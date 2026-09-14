#pragma once

#include "qwen_executor.hpp"
#include <stdexcept>

namespace qw3::detail {

// Dense prefix snapshots are immutable. Cold storage keeps their exact bytes;
// the caller owns KV-page leases and serializes cache policy changes.
class PrefixStateStorage {
public:
    using Snapshot = QwenExecutor::StateSnapshot;
    explicit PrefixStateStorage(DeviceBackend &backend) : backend_(backend) {}
    ~PrefixStateStorage() {
        // An entry may be evicted while its D2H is in flight. Never release
        // source tensors or pinned host memory ahead of that copy.
        if (pending_) {
            if (!backend_.wait_kv_transfer_fence(pending_).ok)
                (void)backend_.synchronize();
        }
    }

    std::shared_ptr<Snapshot> gpu;
    const Snapshot &state() const { return gpu ? *gpu : *layout_; }
    bool copying() const { return copying_; }
    bool has_host_copy() const { return static_cast<bool>(host_); }

    static void copy_metadata(Snapshot &dst, const Snapshot &src) {
        dst.gc_lease = src.gc_lease;
        dst.ready = src.ready;
        dst.position = src.position;
        dst.kv_logical_pages = src.kv_logical_pages;
        dst.mtp_kv_logical_pages = src.mtp_kv_logical_pages;
        dst.mtp_prefix_len = src.mtp_prefix_len;
        dst.mtp_payload_valid_through = src.mtp_payload_valid_through;
        dst.kvmem_registered_pos = src.kvmem_registered_pos;
        dst.kvmem_selection_generation = src.kvmem_selection_generation;
        dst.kvmem_active = src.kvmem_active;
        dst.window_query_pos = src.window_query_pos;
        dst.window_page_count = src.window_page_count;
        dst.window_mrope_next = src.window_mrope_next;
        dst.window_mrope_append_active = src.window_mrope_append_active;
        dst.window_mrope_append_source_next = src.window_mrope_append_source_next;
        dst.window_mrope_append_delta = src.window_mrope_append_delta;
    }

    template<class F> static void tensors(Snapshot &state, F fn) {
        fn(state.h);
        fn(state.mtp_prefix_h);
        for (auto &tensor : state.recurrent_states) fn(tensor);
        for (auto &tensor : state.conv_states) fn(tensor);
    }

    static uint64_t bytes(Snapshot &state) {
        uint64_t total = 0;
        tensors(state, [&](const auto &tensor) {
            if (tensor) total += tensor->count * tensor->elem_size;
        });
        return total;
    }

    void stage_out(std::shared_ptr<HostBuffer> host) {
        if (!gpu) throw std::logic_error("prefix snapshot is already cold");
        if (host_) { copying_ = true; return; } // Promoted snapshots are immutable.
        if (!host || host->bytes < bytes(*gpu))
            throw std::runtime_error("prefix host snapshot buffer is too small");
        auto layout = std::make_unique<Snapshot>();
        copy_metadata(*layout, *gpu);
        layout->recurrent_states.resize(gpu->recurrent_states.size());
        layout->conv_states.resize(gpu->conv_states.size());
        auto metadata = [](const auto &source) -> std::unique_ptr<DeviceTensor> {
            if (!source) return {};
            // These descriptors are never passed to device kernels/copies.
            auto out = std::make_unique<DeviceTensor>();
            out->count = source->count;
            out->dtype = source->dtype;
            out->elem_size = source->elem_size;
            return out;
        };
        layout->h = metadata(gpu->h);
        layout->mtp_prefix_h = metadata(gpu->mtp_prefix_h);
        for (size_t i = 0; i < gpu->recurrent_states.size(); ++i)
            layout->recurrent_states[i] = metadata(gpu->recurrent_states[i]);
        for (size_t i = 0; i < gpu->conv_states.size(); ++i)
            layout->conv_states[i] = metadata(gpu->conv_states[i]);
        try {
            require(backend_.begin_kv_transfer_from_device());
            uint64_t offset = 0;
            tensors(*gpu, [&](const auto &tensor) {
                if (!tensor) return;
                const uint64_t n = tensor->count * tensor->elem_size;
                require(backend_.copy_bytes_to_host_async(*tensor,
                    static_cast<uint8_t *>(host->data) + offset, 0, n));
                offset += n;
            });
            require(backend_.record_kv_transfer_fence(pending_));
        } catch (...) {
            (void)backend_.wait_kv_transfer();
            throw;
        }
        host_ = std::move(host);
        layout_ = std::move(layout);
        copying_ = true;
    }

    // Called after request completion, when the transfer has overlapped decode.
    // A borrowed snapshot cannot be recycled even if its cache entry is idle.
    std::shared_ptr<Snapshot> finish_offload() {
        if (!gpu || !host_ || gpu.use_count() != 1) return {};
        require(backend_.wait_kv_transfer_fence(pending_));
        copying_ = false;
        return std::move(gpu);
    }

    std::shared_ptr<Snapshot> restore(std::shared_ptr<Snapshot> recycled = {}) {
        if (gpu) return gpu;
        if (!host_ || !layout_) throw std::logic_error("prefix snapshot has no backing");
        auto next = recycled ? std::move(recycled) : std::make_shared<Snapshot>();
        copy_metadata(*next, *layout_);
        auto prepare = [&](auto &dst, const auto &src) {
            if (!src) { dst.reset(); return; }
            if (src->dtype != DeviceTensorDType::F32 && src->dtype != DeviceTensorDType::BF16)
                throw std::runtime_error("unsupported prefix snapshot dtype");
            if (!dst || dst->count != src->count || dst->dtype != src->dtype ||
                dst->elem_size != src->elem_size)
                dst = backend_.scratch_like(*src, src->count, "prefix_restore");
            if (dst->dtype != src->dtype || dst->elem_size != src->elem_size)
                throw std::runtime_error("prefix snapshot restoration changed dtype");
        };
        prepare(next->h, layout_->h);
        prepare(next->mtp_prefix_h, layout_->mtp_prefix_h);
        next->recurrent_states.resize(layout_->recurrent_states.size());
        next->conv_states.resize(layout_->conv_states.size());
        for (size_t i = 0; i < next->recurrent_states.size(); ++i)
            prepare(next->recurrent_states[i], layout_->recurrent_states[i]);
        for (size_t i = 0; i < next->conv_states.size(); ++i)
            prepare(next->conv_states[i], layout_->conv_states[i]);
        try {
            // Newly allocated BF16 tensors may still have initialization on
            // the execution stream. Order H2D after it, then wait once for the
            // entire packed snapshot before handing it to the executor.
            require(backend_.begin_kv_transfer_from_device());
            uint64_t offset = 0;
            tensors(*next, [&](auto &tensor) {
                if (!tensor) return;
                const uint64_t n = tensor->count * tensor->elem_size;
                require(backend_.copy_bytes_from_host_async(*tensor, 0,
                    static_cast<const uint8_t *>(host_->data) + offset, n));
                offset += n;
            });
            require(backend_.wait_kv_transfer());
        } catch (...) {
            (void)backend_.wait_kv_transfer();
            throw;
        }
        gpu = std::move(next);
        return gpu;
    }

private:
    static void require(DeviceStatus st) {
        if (!st.ok) throw std::runtime_error(st.message);
    }
    DeviceBackend &backend_;
    std::unique_ptr<Snapshot> layout_;
    std::shared_ptr<HostBuffer> host_;
    std::unique_ptr<DeviceTransferFence> pending_;
    bool copying_ = false;
};

} // namespace qw3::detail
