#pragma once

// NVMe KV tier — host-side slot allocator + raw byte store.
//
// One slot stores one KVMem block's KV bytes across all standard-attention
// layers. This class owns the backing file and synchronous read/write methods;
// executor-level code is responsible for staging through CPU pinned memory and
// for preserving canonical true-position KV semantics.

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

namespace qw3 {

struct NvmeKvTierConfig {
    std::string dir;
    uint64_t total_bytes = 0;
    uint64_t slot_bytes = 0;
};

struct NvmeSlotPlacement {
    int32_t slot = -1;
    int32_t evicted_block = -1;
};

class NvmeKvTier {
public:
    explicit NvmeKvTier(NvmeKvTierConfig cfg) : cfg_(std::move(cfg)) {
        if (cfg_.slot_bytes > 0 && cfg_.total_bytes >= cfg_.slot_bytes) {
            slot_count_ = static_cast<uint32_t>(cfg_.total_bytes / cfg_.slot_bytes);
        }
        if (slot_count_ == 0 || cfg_.dir.empty()) return;

        ensure_dir(cfg_.dir);
        path_ = cfg_.dir + "/qw3_kvmem_nvme.bin";
        file_ = std::fopen(path_.c_str(), "w+b");
        if (!file_) {
            throw std::runtime_error("failed to open NVMe KV tier file: " + path_);
        }
        free_slots_.reserve(slot_count_);
        for (uint32_t i = 0; i < slot_count_; ++i) {
            free_slots_.push_back(static_cast<int32_t>(slot_count_ - 1U - i));
        }
    }

    NvmeKvTier(const NvmeKvTier &) = delete;
    NvmeKvTier &operator=(const NvmeKvTier &) = delete;

    ~NvmeKvTier() {
        if (file_) std::fclose(file_);
    }

    bool enabled() const { return file_ != nullptr && slot_count_ > 0; }
    uint32_t slot_count() const { return slot_count_; }
    uint32_t free_slots() const { return static_cast<uint32_t>(free_slots_.size()); }
    uint32_t used_slots() const { return slot_count_ - free_slots(); }
    const std::string &path() const { return path_; }

    uint64_t slot_offset(int32_t slot) const {
        return static_cast<uint64_t>(slot) * cfg_.slot_bytes;
    }

    int32_t block_slot(uint32_t block_id) const {
        auto it = block_to_slot_.find(block_id);
        return it == block_to_slot_.end() ? -1 : it->second;
    }

    NvmeSlotPlacement place_block(uint32_t block_id) {
        NvmeSlotPlacement out;
        auto it = block_to_slot_.find(block_id);
        if (it != block_to_slot_.end()) {
            out.slot = it->second;
            touch(block_id);
            return out;
        }
        if (free_slots_.empty()) return out;
        const int32_t slot = free_slots_.back();
        free_slots_.pop_back();
        block_to_slot_[block_id] = slot;
        touch(block_id);
        out.slot = slot;
        return out;
    }

    NvmeSlotPlacement place_block_evicting(uint32_t block_id) {
        NvmeSlotPlacement out = place_block(block_id);
        if (out.slot >= 0 || !free_slots_.empty()) return out;
        if (lru_.empty()) return out;
        const uint32_t victim = lru_.front();
        auto vit = block_to_slot_.find(victim);
        if (vit == block_to_slot_.end()) {
            erase_lru(victim);
            return out;
        }
        const int32_t slot = vit->second;
        block_to_slot_.erase(vit);
        erase_lru(victim);
        block_to_slot_[block_id] = slot;
        touch(block_id);
        out.slot = slot;
        out.evicted_block = static_cast<int32_t>(victim);
        return out;
    }

    void release_block(uint32_t block_id) {
        auto it = block_to_slot_.find(block_id);
        if (it == block_to_slot_.end()) return;
        free_slots_.push_back(it->second);
        block_to_slot_.erase(it);
        erase_lru(block_id);
    }

    void clear() {
        free_slots_.clear();
        for (uint32_t i = 0; i < slot_count_; ++i) {
            free_slots_.push_back(static_cast<int32_t>(slot_count_ - 1U - i));
        }
        block_to_slot_.clear();
        lru_.clear();
    }

    void touch(uint32_t block_id) {
        erase_lru(block_id);
        lru_.push_back(block_id);
    }

    int32_t lru_victim() const {
        return lru_.empty() ? -1 : static_cast<int32_t>(lru_.front());
    }

    void write_block(uint32_t block_id, const void *data, uint64_t bytes) {
        if (!enabled()) throw std::runtime_error("NVMe KV tier is disabled");
        if (!data && bytes > 0) throw std::runtime_error("NVMe write null data");
        if (bytes > cfg_.slot_bytes) {
            throw std::runtime_error("NVMe write exceeds slot size");
        }
        auto p = place_block(block_id);
        if (p.slot < 0) {
            throw std::runtime_error("NVMe KV tier is full");
        }
        write_slot(p.slot, data, bytes);
        touch(block_id);
    }

    void read_block(uint32_t block_id, void *data, uint64_t bytes) {
        if (!enabled()) throw std::runtime_error("NVMe KV tier is disabled");
        if (!data && bytes > 0) throw std::runtime_error("NVMe read null data");
        if (bytes > cfg_.slot_bytes) {
            throw std::runtime_error("NVMe read exceeds slot size");
        }
        const int32_t slot = block_slot(block_id);
        if (slot < 0) throw std::runtime_error("NVMe block is not resident");
        read_slot(slot, data, bytes);
        touch(block_id);
    }

private:
    static void ensure_dir(const std::string &dir) {
        struct stat st {};
        if (stat(dir.c_str(), &st) == 0) {
            if ((st.st_mode & S_IFDIR) == 0) {
                throw std::runtime_error("NVMe KV tier path is not a directory: " + dir);
            }
            return;
        }
        if (mkdir(dir.c_str(), 0755) != 0) {
            throw std::runtime_error("failed to create NVMe KV tier directory: " + dir);
        }
    }

    void write_slot(int32_t slot, const void *data, uint64_t bytes) {
        if (std::fseek(file_, static_cast<long>(slot_offset(slot)), SEEK_SET) != 0) {
            throw std::runtime_error("NVMe seek failed on write");
        }
        if (bytes > 0 && std::fwrite(data, 1, static_cast<size_t>(bytes), file_) != bytes) {
            throw std::runtime_error("NVMe write failed");
        }
        std::fflush(file_);
    }

    void read_slot(int32_t slot, void *data, uint64_t bytes) {
        if (std::fseek(file_, static_cast<long>(slot_offset(slot)), SEEK_SET) != 0) {
            throw std::runtime_error("NVMe seek failed on read");
        }
        if (bytes > 0 && std::fread(data, 1, static_cast<size_t>(bytes), file_) != bytes) {
            throw std::runtime_error("NVMe read failed");
        }
    }

    void erase_lru(uint32_t block_id) {
        for (auto it = lru_.begin(); it != lru_.end(); ++it) {
            if (*it == block_id) {
                lru_.erase(it);
                return;
            }
        }
    }

    NvmeKvTierConfig cfg_;
    uint32_t slot_count_ = 0;
    std::string path_;
    std::FILE *file_ = nullptr;
    std::vector<int32_t> free_slots_;
    std::unordered_map<uint32_t, int32_t> block_to_slot_;
    std::vector<uint32_t> lru_;
};

} // namespace qw3
