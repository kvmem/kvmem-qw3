#include "prefix_state_storage.hpp"
#include <cstring>
#include <iostream>
#include <vector>

using namespace qw3;
using Storage = detail::PrefixStateStorage;

static void check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
static void ok(DeviceStatus status) { check(status.ok, status.message); }

int main() {
    try {
        auto backend = make_cuda_device_backend(LinearBackend::Auto);
        check(static_cast<bool>(backend), "CUDA backend unavailable");
        ok(backend->begin());
        {
            Storage stored(*backend);
            stored.gpu = std::make_shared<Storage::Snapshot>();
            auto &s = *stored.gpu;
            s.ready = true;
            s.position = 96;
            s.kv_logical_pages = 6;
            s.mtp_kv_logical_pages = 6;
            s.mtp_prefix_len = 96;
            s.mtp_payload_valid_through = 100;
            s.kvmem_registered_pos = 95;
            s.kvmem_selection_generation = 123;
            s.kvmem_active = true;
            s.window_query_pos = 32;
            s.window_page_count = 2;
            s.window_mrope_next = 44;
            s.window_mrope_append_active = true;
            s.window_mrope_append_source_next = 77;
            s.window_mrope_append_delta = {-6, 7, 8};
            s.h = backend->tensor_bf16(15, "test hidden");
            s.mtp_prefix_h = backend->tensor_f32(17, "test MTP hidden");
            s.recurrent_states.resize(3);
            s.conv_states.resize(3);
            s.recurrent_states[0] = backend->tensor_f32(256, "test recurrent");
            s.recurrent_states[2] = backend->tensor_f32(256, "test recurrent 2");
            s.conv_states[0] = backend->tensor_f32(13, "test convolution");
            s.conv_states[2] = backend->tensor_f32(13, "test convolution 2");
            std::vector<std::vector<uint8_t>> originals;
            Storage::tensors(s, [&](auto &tensor) {
                if (!tensor) return;
                std::vector<uint8_t> data(tensor->count * tensor->elem_size);
                // Raw bit patterns exercise NaNs/subnormals and BF16 without
                // relying on numeric comparisons that could hide conversion.
                for (size_t i = 0; i < data.size(); ++i)
                    data[i] = static_cast<uint8_t>(i * 17 + originals.size() * 29);
                ok(backend->copy_bytes_from_host(*tensor, 0, data.data(), data.size()));
                originals.push_back(std::move(data));
            });
            std::shared_ptr<HostBuffer> host = backend->host_buffer(Storage::bytes(s), "test packed snapshot");
            stored.stage_out(host);
            auto reader = stored.gpu;
            check(!stored.finish_offload(), "in-flight reader was recycled");
            reader.reset();
            auto recycled = stored.finish_offload();
            check(recycled && !stored.gpu, "snapshot was not demoted");
            // Overwrite recycled GPU backing before restoring: CPU data must
            // be authoritative and cannot silently alias recycled device data.
            Storage::tensors(*recycled, [&](auto &tensor) {
                if (!tensor) return;
                std::vector<uint8_t> zeros(tensor->count * tensor->elem_size, 0);
                ok(backend->copy_bytes_from_host(*tensor, 0, zeros.data(), zeros.size()));
            });
            auto restored = stored.restore(std::move(recycled));
            check(restored->ready && restored->position == 96 &&
                  restored->kv_logical_pages == 6 && restored->mtp_kv_logical_pages == 6 &&
                  restored->mtp_prefix_len == 96 && restored->mtp_payload_valid_through == 100 &&
                  restored->kvmem_registered_pos == 95 && restored->kvmem_selection_generation == 123 &&
                  restored->kvmem_active && restored->window_query_pos == 32 &&
                  restored->window_page_count == 2 && restored->window_mrope_next == 44 &&
                  restored->window_mrope_append_active && restored->window_mrope_append_source_next == 77 &&
                  restored->window_mrope_append_delta == std::array<int64_t, 3>{-6, 7, 8},
                  "snapshot metadata changed");
            check(!restored->recurrent_states[1] && !restored->conv_states[1], "empty layers changed");
            size_t index = 0;
            Storage::tensors(*restored, [&](auto &tensor) {
                if (!tensor) return;
                std::vector<uint8_t> data(tensor->count * tensor->elem_size);
                ok(backend->copy_bytes_to_host(*tensor, data.data(), 0, data.size()));
                check(data == originals.at(index++), "snapshot bytes changed");
            });
            check(index == originals.size(), "tensor count changed");
            restored.reset();
            stored.stage_out({}); // Existing immutable CPU backing is reused.
            auto again = stored.finish_offload();
            check(again && !stored.gpu, "promoted snapshot did not reuse host backing");
            stored.restore(std::move(again));
        }
        {
            Storage pending(*backend);
            pending.gpu = std::make_shared<Storage::Snapshot>();
            pending.gpu->h = backend->tensor_f32(4096, "pending source");
            std::shared_ptr<HostBuffer> host = backend->host_buffer(16384, "pending target");
            pending.stage_out(std::move(host));
            // Destruction waits before either backing allocation is released.
        }
        ok(backend->end());
        std::cout << "PASS: bitwise snapshot round-trip, metadata, reader lifetime, recycling, pending destruction\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
