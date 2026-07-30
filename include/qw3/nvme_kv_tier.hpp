#pragma once

// NVMe KV tier — fixed-slot metadata plus positional byte I/O.
//
// The legacy API remains synchronous. Positional pread/pwrite removes the
// shared FILE* cursor so stage-out writes and stage-in reads may safely run on
// different host workers. Batch spans coalesce adjacent full records into one
// syscall when both their file offsets and buffer ranges are contiguous.

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace qw3 {

struct NvmeKvTierConfig {
    std::string dir;
    // Separate logical tiers may share one directory while retaining
    // independent anonymous backing files. The directory entry is unlinked
    // immediately after open, so this is a collision-avoidance/debug label,
    // not a persistent cache name.
    std::string file_name = "qw3_kvmem_nvme.bin";
    uint64_t total_bytes = 0;
    uint64_t slot_bytes = 0;
    // Buffered I/O otherwise keeps a second copy of the SSD arena in the
    // kernel page cache.  For long contexts that copy can be tens of GiB and
    // defeats the explicit CPU-tier memory budget.  When enabled, completed
    // writes are range-written back and both reads and writes are advised
    // DONTNEED after the user buffer owns the data.
    bool drop_page_cache = false;
};

struct NvmeSlotPlacement {
    int32_t slot = -1;
    int32_t evicted_block = -1;
};

struct NvmeIoSpan {
    int32_t slot = -1;
    uint64_t buffer_offset = 0;
    uint64_t bytes = 0;
};

struct NvmeBatchIoStats {
    uint64_t bytes = 0;
    uint64_t syscalls = 0;
    uint64_t duration_ns = 0;
    uint64_t cache_drop_bytes = 0;
    uint64_t cache_drop_failures = 0;
    uint64_t cpu_copy_bytes = 0;
    uint64_t cpu_copy_ns = 0;
    uint32_t cpu_copy_blocks = 0;
};

class NvmeKvTier {
public:
    explicit NvmeKvTier(NvmeKvTierConfig cfg) : cfg_(std::move(cfg)) {
        if (cfg_.slot_bytes > 0 && cfg_.total_bytes >= cfg_.slot_bytes) {
            slot_count_ = static_cast<uint32_t>(
                cfg_.total_bytes / cfg_.slot_bytes);
        }
        if (slot_count_ == 0 || cfg_.dir.empty()) return;

        ensure_dir(cfg_.dir);
        if (cfg_.file_name.empty() ||
            cfg_.file_name.find('/') != std::string::npos) {
            throw std::runtime_error(
                "NVMe KV tier file_name must be a non-empty basename");
        }
        path_ = cfg_.dir + "/" + cfg_.file_name;
        fd_ = ::open(path_.c_str(), O_CREAT | O_TRUNC | O_RDWR | O_CLOEXEC,
                     0644);
        if (fd_ < 0) {
            throw std::runtime_error(
                "failed to open NVMe KV tier file: " + path_ + ": " +
                std::strerror(errno));
        }
        // The backing store is an ephemeral cache, never a recoverable
        // checkpoint. Unlink it immediately while retaining the open file
        // descriptor: all positional I/O continues to work, but the filesystem
        // reclaims the blocks automatically when the process closes the fd,
        // including abnormal exits and SIGKILL. This also prevents stale
        // qw3_kvmem_nvme.bin files from accumulating across evaluations.
        if (::unlink(path_.c_str()) != 0) {
            const int unlink_error = errno;
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error(
                "failed to make NVMe KV tier file ephemeral: " + path_ +
                ": " + std::strerror(unlink_error));
        }
        free_slots_.reserve(slot_count_);
        for (uint32_t i = 0; i < slot_count_; ++i) {
            free_slots_.push_back(
                static_cast<int32_t>(slot_count_ - 1U - i));
        }
    }

    NvmeKvTier(const NvmeKvTier &) = delete;
    NvmeKvTier &operator=(const NvmeKvTier &) = delete;

    ~NvmeKvTier() {
        if (fd_ >= 0) ::close(fd_);
    }

    bool enabled() const { return fd_ >= 0 && slot_count_ > 0; }
    uint32_t slot_count() const { return slot_count_; }
    uint64_t slot_bytes() const { return cfg_.slot_bytes; }
    const std::string &path() const { return path_; }
    bool drops_page_cache() const { return cfg_.drop_page_cache; }

    uint32_t free_slots() const {
        std::lock_guard<std::mutex> lock(meta_mu_);
        return static_cast<uint32_t>(free_slots_.size());
    }

    uint32_t used_slots() const {
        return slot_count_ - free_slots();
    }

    uint64_t slot_offset(int32_t slot) const {
        return static_cast<uint64_t>(slot) * cfg_.slot_bytes;
    }

    int32_t block_slot(uint32_t block_id) const {
        std::lock_guard<std::mutex> lock(meta_mu_);
        auto it = block_to_slot_.find(block_id);
        return it == block_to_slot_.end() ? -1 : it->second;
    }

    NvmeSlotPlacement place_block(uint32_t block_id) {
        std::lock_guard<std::mutex> lock(meta_mu_);
        return place_block_locked(block_id);
    }

    NvmeSlotPlacement place_block_evicting(uint32_t block_id) {
        std::lock_guard<std::mutex> lock(meta_mu_);
        NvmeSlotPlacement out = place_block_locked(block_id);
        if (out.slot >= 0 || !free_slots_.empty()) return out;
        if (lru_.empty()) return out;
        const uint32_t victim = lru_.front();
        auto vit = block_to_slot_.find(victim);
        if (vit == block_to_slot_.end()) {
            erase_lru_locked(victim);
            return out;
        }
        const int32_t slot = vit->second;
        block_to_slot_.erase(vit);
        erase_lru_locked(victim);
        block_to_slot_[block_id] = slot;
        touch_locked(block_id);
        out.slot = slot;
        out.evicted_block = static_cast<int32_t>(victim);
        return out;
    }

    void release_block(uint32_t block_id) {
        std::lock_guard<std::mutex> lock(meta_mu_);
        auto it = block_to_slot_.find(block_id);
        if (it == block_to_slot_.end()) return;
        free_slots_.push_back(it->second);
        block_to_slot_.erase(it);
        erase_lru_locked(block_id);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(meta_mu_);
        free_slots_.clear();
        for (uint32_t i = 0; i < slot_count_; ++i) {
            free_slots_.push_back(
                static_cast<int32_t>(slot_count_ - 1U - i));
        }
        block_to_slot_.clear();
        lru_.clear();
    }

    void touch(uint32_t block_id) {
        std::lock_guard<std::mutex> lock(meta_mu_);
        touch_locked(block_id);
    }

    int32_t lru_victim() const {
        std::lock_guard<std::mutex> lock(meta_mu_);
        return lru_.empty() ? -1 : static_cast<int32_t>(lru_.front());
    }

    void write_block(uint32_t block_id, const void *data, uint64_t bytes) {
        validate_io(data, bytes, "write");
        auto p = place_block(block_id);
        if (p.slot < 0) {
            throw std::runtime_error("NVMe KV tier is full");
        }
        write_slot(p.slot, data, bytes);
        touch(block_id);
    }

    void read_block(uint32_t block_id, void *data, uint64_t bytes) {
        validate_io(data, bytes, "read");
        const int32_t slot = block_slot(block_id);
        if (slot < 0) {
            throw std::runtime_error("NVMe block is not resident");
        }
        read_slot(slot, data, bytes);
        touch(block_id);
    }

    // The slot must already have been reserved with place_block(). These raw
    // positional methods do not touch metadata and are safe on worker threads.
    void write_slot(int32_t slot, const void *data, uint64_t bytes) const {
        write_slot_range(slot, 0, data, bytes);
    }

    void write_slot_range(int32_t slot, uint64_t slot_byte_offset,
                          const void *data, uint64_t bytes) const {
        validate_slot_range_io(
            slot, slot_byte_offset, data, bytes, "write");
        const uint64_t offset = slot_offset(slot) + slot_byte_offset;
        pwrite_all(data, bytes, offset);
        if (cfg_.drop_page_cache) {
            (void) drop_cached_range(offset, bytes, /*write=*/true);
        }
    }

    void read_slot(int32_t slot, void *data, uint64_t bytes) const {
        read_slot_range(slot, 0, data, bytes);
    }

    void read_slot_range(int32_t slot, uint64_t slot_byte_offset,
                         void *data, uint64_t bytes) const {
        validate_slot_range_io(
            slot, slot_byte_offset, data, bytes, "read");
        const uint64_t offset = slot_offset(slot) + slot_byte_offset;
        pread_all(data, bytes, offset);
        if (cfg_.drop_page_cache) {
            (void) drop_cached_range(offset, bytes, /*write=*/false);
        }
    }

    void write_spans(const std::vector<NvmeIoSpan> &spans,
                     const void *buffer, uint64_t buffer_bytes,
                     NvmeBatchIoStats *stats = nullptr) const {
        run_spans(spans, const_cast<void *>(buffer), buffer_bytes,
                  /*write=*/true, stats);
    }

    void read_spans(const std::vector<NvmeIoSpan> &spans,
                    void *buffer, uint64_t buffer_bytes,
                    NvmeBatchIoStats *stats = nullptr) const {
        run_spans(spans, buffer, buffer_bytes, /*write=*/false, stats);
    }

private:
    static void ensure_dir(const std::string &dir) {
        struct stat st {};
        if (stat(dir.c_str(), &st) == 0) {
            if ((st.st_mode & S_IFDIR) == 0) {
                throw std::runtime_error(
                    "NVMe KV tier path is not a directory: " + dir);
            }
            return;
        }
        if (mkdir(dir.c_str(), 0755) != 0) {
            throw std::runtime_error(
                "failed to create NVMe KV tier directory: " + dir);
        }
    }

    void validate_io(const void *data, uint64_t bytes,
                     const char *op) const {
        if (!enabled()) {
            throw std::runtime_error("NVMe KV tier is disabled");
        }
        if (!data && bytes > 0) {
            throw std::runtime_error(
                std::string("NVMe ") + op + " null data");
        }
        if (bytes > cfg_.slot_bytes) {
            throw std::runtime_error(
                std::string("NVMe ") + op + " exceeds slot size");
        }
    }

    void validate_slot_io(int32_t slot, const void *data, uint64_t bytes,
                          const char *op) const {
        validate_io(data, bytes, op);
        if (slot < 0 || static_cast<uint32_t>(slot) >= slot_count_) {
            throw std::runtime_error(
                std::string("NVMe ") + op + " has invalid slot");
        }
    }

    void validate_slot_range_io(int32_t slot, uint64_t slot_byte_offset,
                                const void *data, uint64_t bytes,
                                const char *op) const {
        validate_slot_io(slot, data, bytes, op);
        if (slot_byte_offset > cfg_.slot_bytes ||
            bytes > cfg_.slot_bytes - slot_byte_offset) {
            throw std::runtime_error(
                std::string("NVMe ") + op + " range exceeds slot size");
        }
    }

    NvmeSlotPlacement place_block_locked(uint32_t block_id) {
        NvmeSlotPlacement out;
        auto it = block_to_slot_.find(block_id);
        if (it != block_to_slot_.end()) {
            out.slot = it->second;
            touch_locked(block_id);
            return out;
        }
        if (free_slots_.empty()) return out;
        const int32_t slot = free_slots_.back();
        free_slots_.pop_back();
        block_to_slot_[block_id] = slot;
        touch_locked(block_id);
        out.slot = slot;
        return out;
    }

    void pwrite_all(const void *data, uint64_t bytes, uint64_t offset) const {
        const uint8_t *src = static_cast<const uint8_t *>(data);
        uint64_t done = 0;
        while (done < bytes) {
            const ssize_t n = ::pwrite(
                fd_, src + done, static_cast<size_t>(bytes - done),
                static_cast<off_t>(offset + done));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) {
                throw std::runtime_error(
                    "NVMe positional write failed: " +
                    std::string(std::strerror(errno)));
            }
            done += static_cast<uint64_t>(n);
        }
    }

    void pread_all(void *data, uint64_t bytes, uint64_t offset) const {
        uint8_t *dst = static_cast<uint8_t *>(data);
        uint64_t done = 0;
        while (done < bytes) {
            const ssize_t n = ::pread(
                fd_, dst + done, static_cast<size_t>(bytes - done),
                static_cast<off_t>(offset + done));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) {
                throw std::runtime_error(
                    "NVMe positional read failed or reached unwritten data");
            }
            done += static_cast<uint64_t>(n);
        }
    }

    void run_spans(const std::vector<NvmeIoSpan> &spans, void *buffer,
                   uint64_t buffer_bytes, bool write,
                   NvmeBatchIoStats *stats) const {
        if (stats) *stats = NvmeBatchIoStats{};
        if (spans.empty()) return;
        if (!buffer) throw std::runtime_error("NVMe batch I/O null buffer");
        uint8_t *base = static_cast<uint8_t *>(buffer);
        size_t i = 0;
        while (i < spans.size()) {
            const NvmeIoSpan &first = spans[i];
            validate_slot_io(first.slot, base + first.buffer_offset,
                             first.bytes, write ? "write" : "read");
            if (first.buffer_offset + first.bytes > buffer_bytes) {
                throw std::runtime_error("NVMe batch span exceeds buffer");
            }
            uint64_t merged_bytes = first.bytes;
            size_t j = i + 1;
            while (j < spans.size()) {
                const NvmeIoSpan &prev = spans[j - 1];
                const NvmeIoSpan &next = spans[j];
                const bool contiguous_file =
                    next.slot == prev.slot + 1 &&
                    prev.bytes == cfg_.slot_bytes;
                const bool contiguous_buffer =
                    next.buffer_offset ==
                    spans[i].buffer_offset + merged_bytes;
                if (!contiguous_file || !contiguous_buffer) break;
                validate_slot_io(next.slot, base + next.buffer_offset,
                                 next.bytes, write ? "write" : "read");
                if (next.buffer_offset + next.bytes > buffer_bytes) {
                    throw std::runtime_error(
                        "NVMe batch span exceeds buffer");
                }
                merged_bytes += next.bytes;
                ++j;
            }
            if (write) {
                pwrite_all(base + first.buffer_offset, merged_bytes,
                           slot_offset(first.slot));
            } else {
                pread_all(base + first.buffer_offset, merged_bytes,
                          slot_offset(first.slot));
            }
            if (cfg_.drop_page_cache) {
                const uint64_t file_offset = slot_offset(first.slot);
                const bool dropped =
                    drop_cached_range(file_offset, merged_bytes, write);
                if (stats) {
                    if (dropped) {
                        stats->cache_drop_bytes += merged_bytes;
                    } else {
                        ++stats->cache_drop_failures;
                    }
                }
            }
            if (stats) {
                stats->bytes += merged_bytes;
                ++stats->syscalls;
            }
            i = j;
        }
    }

    bool drop_cached_range(uint64_t offset, uint64_t bytes,
                           bool write) const {
        if (bytes == 0) return true;
        if (write) {
#if defined(__linux__) && defined(SYNC_FILE_RANGE_WRITE) && \
    defined(SYNC_FILE_RANGE_WAIT_BEFORE) && \
    defined(SYNC_FILE_RANGE_WAIT_AFTER)
            int rc;
            do {
                rc = ::sync_file_range(
                    fd_, static_cast<off64_t>(offset),
                    static_cast<off64_t>(bytes),
                    SYNC_FILE_RANGE_WAIT_BEFORE |
                        SYNC_FILE_RANGE_WRITE |
                        SYNC_FILE_RANGE_WAIT_AFTER);
            } while (rc != 0 && errno == EINTR);
            if (rc != 0) {
                warn_cache_drop_failure("range-writeback", errno);
                return false;
            }
#else
            // Portable fallback. It flushes the whole file rather than one
            // range, so serialize concurrent batches to avoid redundant
            // fdatasync storms. Linux uses sync_file_range above.
            std::lock_guard<std::mutex> lock(cache_drop_mu_);
            int rc;
            do {
                rc = ::fdatasync(fd_);
            } while (rc != 0 && errno == EINTR);
            if (rc != 0) {
                warn_cache_drop_failure("fdatasync", errno);
                return false;
            }
#endif
        }
#if defined(POSIX_FADV_DONTNEED)
        const int advise = ::posix_fadvise(
            fd_, static_cast<off_t>(offset), static_cast<off_t>(bytes),
            POSIX_FADV_DONTNEED);
        if (advise != 0) {
            warn_cache_drop_failure("posix-fadvise-dontneed", advise);
        }
        return advise == 0;
#else
        (void) offset;
        warn_cache_drop_failure("posix-fadvise-unavailable", ENOTSUP);
        return false;
#endif
    }

    void warn_cache_drop_failure(const char *phase, int error) const {
        bool expected = false;
        if (!cache_drop_warned_.compare_exchange_strong(expected, true)) {
            return;
        }
        std::fprintf(
            stderr,
            "[kvmem-io] page_cache_drop_degraded=1 phase=%s error=%d "
            "message=%s action=continue-with-kernel-page-cache\n",
            phase, error, std::strerror(error));
    }

    void touch_locked(uint32_t block_id) {
        erase_lru_locked(block_id);
        lru_.push_back(block_id);
    }

    void erase_lru_locked(uint32_t block_id) {
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
    int fd_ = -1;
    mutable std::mutex meta_mu_;
    mutable std::mutex cache_drop_mu_;
    mutable std::atomic<bool> cache_drop_warned_{false};
    std::vector<int32_t> free_slots_;
    std::unordered_map<uint32_t, int32_t> block_to_slot_;
    std::vector<uint32_t> lru_;
};

} // namespace qw3
