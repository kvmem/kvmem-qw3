#pragma once

#include "qw3/device_backend.hpp"
#include <functional>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace qw3::detail {
// Recycle only scratch bytes whose last transfer/user has completed. A lease
// has one exclusive backing allocation, including when held by shared_ptr.
// Idle storage is bounded across requests; live storage belongs to each
// request's existing bounded I/O queues. The backend must outlive all leases.
class HostIoBufferPool {
public:
    using Allocate = std::function<std::unique_ptr<HostBuffer>(uint64_t, const char *)>;
    using Drain = std::function<bool()>;
    struct Stats { uint64_t allocations = 0, reuses = 0, cached_bytes = 0; };
private:
    struct State {
        Allocate allocate;
        Drain drain;
        uint64_t limit;
        std::mutex mutex;
        Stats stats;
        std::unordered_map<uint64_t, std::vector<std::unique_ptr<HostBuffer>>> idle;
        State(Allocate fn, uint64_t cap, Drain done)
            : allocate(std::move(fn)), drain(std::move(done)), limit(cap) {}
        void release(std::unique_ptr<HostBuffer> buffer) noexcept {
            if (!buffer) return;
            // Recycling is optional, including under host allocation failure.
            try {
                // cudaFreeHost used to supply an implicit device wait, even
                // on exception paths. Preserve transfer completion explicitly
                // before another request can overwrite these bytes.
                if (drain && !drain()) return;
                std::lock_guard<std::mutex> lock(mutex);
                if (buffer->bytes <= limit - stats.cached_bytes) {
                    const uint64_t bytes = buffer->bytes;
                    idle[bytes].push_back(std::move(buffer));
                    stats.cached_bytes += bytes;
                }
            } catch (...) {}
        }
    };
    struct Lease final : HostBuffer {
        std::weak_ptr<State> pool;
        std::unique_ptr<HostBuffer> storage;
        Lease(const std::shared_ptr<State> &state, std::unique_ptr<HostBuffer> buffer)
            : pool(state), storage(std::move(buffer)) {
            data = storage->data; bytes = storage->bytes; pinned = storage->pinned;
        }
        ~Lease() override {
            if (auto state = pool.lock()) state->release(std::move(storage));
        }
    };
    std::shared_ptr<State> state_;
public:
    HostIoBufferPool(Allocate allocate, uint64_t max_idle_bytes, Drain drain = {})
        : state_(std::make_shared<State>(std::move(allocate), max_idle_bytes, std::move(drain))) {}
    std::unique_ptr<HostBuffer> acquire(uint64_t bytes, const char *label) {
        if (!bytes) return nullptr;
        std::unique_ptr<HostBuffer> storage;
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            auto it = state_->idle.find(bytes);
            if (it != state_->idle.end() && !it->second.empty()) {
                storage = std::move(it->second.back()); it->second.pop_back();
                if (it->second.empty()) state_->idle.erase(it);
                state_->stats.cached_bytes -= bytes;
                ++state_->stats.reuses;
            }
        }
        if (!storage) {
            storage = state_->allocate(bytes, label);
            if (!storage || storage->bytes != bytes || !storage->data)
                throw std::runtime_error("invalid pooled host I/O allocation");
            std::lock_guard<std::mutex> lock(state_->mutex);
            ++state_->stats.allocations;
        }
        return std::make_unique<Lease>(state_, std::move(storage));
    }
    Stats stats() const {
        std::lock_guard<std::mutex> lock(state_->mutex);
        return state_->stats;
    }
};
} // namespace qw3::detail
