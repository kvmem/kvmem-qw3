#include "qwen_executor.hpp"
#include "prefix_state_storage.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <unistd.h>

using namespace qw3;
static void check(bool yes, const char *why) { if (!yes) throw std::runtime_error(why); }
static void ok(DeviceStatus s) { check(s.ok, s.message); }

// Only index/transfer kernels run. Dummy weights provide model metadata, never
// an end-to-end inference result or a model-quality measurement.
struct IndexSource final : ModelSource {
    std::string dir;
    ModelSourceInfo info_{"index-test", "qwen35", 2, 4};
    mutable std::map<std::string, ModelTensorInfo> tensors;
    float value = 0;
    explicit IndexSource(std::string d) : dir(std::move(d)) {}
    const ModelTensorInfo *find_tensor(const std::string &name) const override {
        auto &t = tensors[name]; t.name = name; t.source_name = name;
        t.type = ModelTensorType::F32; t.dims = {1}; t.data = &value; t.bytes = sizeof(value);
        return &t;
    }
    const ModelSourceInfo &info() const override { return info_; }
    const std::string &model_directory() const override { return dir; }
};

namespace qw3 {
struct KvPageTableTestAccess {
    static void query_tiles(const QwenNativeModel &model, const QwenWeights &weights,
                            DeviceBackend &backend, bool fp16) {
        constexpr uint32_t P = 512, B = 32, D = 4, L = 2, Q = 521, S = Q + 7;
        QwenExecutor e(model, weights, backend, P * B);
        KvMemStoreConfig cfg; cfg.block_tokens = B; cfg.repository_blocks = P;
        cfg.index_placement = KvMemIndexPlacement::CPU;
        cfg.index_staging_bytes = 64 * D * (fp16 ? 2 : 4);
        auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{P, B});
        e.block_store_ = std::make_unique<KvMemStore>(cfg, repo);
        e.kvmem_qc_num_layers_ = L; e.kvmem_qc_n_subblocks_ = 1;
        e.kvmem_qc_layer_stride_blocks_ = P;
        e.kvmem_repository_index_.resize(P);
        e.kvmem_prepare_cpu_mean_index(P * L * D, L, P, false);
        e.kvmem_repository_score_frame_.enabled.resize(P);
        for (uint32_t b = 0; b < P; ++b) {
            e.kvmem_repository_index_[b].tokens = B;
            // Include a completely empty index tile and scattered holes.
            e.kvmem_repository_score_frame_.enabled[b] = b >= 64 && b % 5 != 0;
        }
        const auto value = [](uint32_t i) { return (i % 2 ? -1.0f : 1.0f) * std::ldexp(1.0f, int(i % 4) - 3); };
        const auto half = [](uint32_t i) { return uint16_t((i % 2 ? 0x8000 : 0) | ((12 + i % 4) << 10)); };
        for (uint32_t l = 0; l < L; ++l) {
            auto *dst = const_cast<uint8_t *>(e.kvmem_mean_index_host_layer_data(l));
            for (uint32_t i = 0; i < P * D; ++i) {
                if (fp16) { const auto v = half(i + i/9 + l*3); std::memcpy(dst + i*2, &v, 2); }
                else { const auto v = value(i + i/9 + l*3); std::memcpy(dst + i*4, &v, 4); }
            }
        }
        e.g_query_multi_ = fp16 ? backend.tensor_f16(L*S*D, "query_tile_test")
                               : backend.tensor_f32(L*S*D, "query_tile_test");
        e.g_query_multi_count_ = Q;
        std::vector<float> q(L*S*D); std::vector<uint16_t> qh(L*S*D);
        for (uint32_t i = 0; i < q.size(); ++i) { q[i] = value(i + i/D + i/37); qh[i] = half(i + i/D + i/37); }
        ok(backend.copy_bytes_from_host(*e.g_query_multi_, 0,
            fp16 ? static_cast<void *>(qh.data()) : static_cast<void *>(q.data()), q.size() * (fp16 ? 2 : 4)));
        e.g_score_dev_ = backend.tensor_f32(P, "query_tile_scores");
        std::vector<float> tiled(P), full(P);
        std::string reason;
        setenv("QW3_KVMEM_ADAPTIVE_BLOCK_STATS_MIB", "1", 1);
        check(e.kvmem_score_cpu_prototype_index(P,Q,S,0.5f,0,P,0,&reason), reason.c_str());
        ok(backend.copy_to_host(*e.g_score_dev_, tiled.data(), 0, P));
        check(e.g_adaptive_stream_block_stats_capacity_ * 8 <= 1024*1024,
              "query tiling exceeded statistics workspace cap");
        setenv("QW3_KVMEM_ADAPTIVE_BLOCK_STATS_MIB", "8", 1);
        check(e.kvmem_score_cpu_prototype_index(P,Q,S,0.5f,0,P,0,&reason), reason.c_str());
        ok(backend.copy_to_host(*e.g_score_dev_, full.data(), 0, P));
        double mass = 0;
        for (uint32_t b = 0; b < P; ++b) {
            check(std::isfinite(tiled[b]) && std::abs(tiled[b] - full[b]) < 1e-4f,
                  "Q-tiled score differs from complete-query scoring");
            if (!e.kvmem_repository_score_frame_.enabled[b]) check(tiled[b] == 0, "deleted slot received score");
            mass += tiled[b];
        }
        check(std::abs(mass - Q) < 1e-2, "query rows were dropped or renormalized incorrectly");
        unsetenv("QW3_KVMEM_ADAPTIVE_BLOCK_STATS_MIB");
        std::cout << (fp16 ? "FP16/GEMM" : "FP32/scalar") << " Q tiling: all 521 rows, masked holes, unequal final tile and layer stride passed\n";
    }
    static void legacy_tail(const QwenNativeModel &model, const QwenWeights &weights,
                            DeviceBackend &backend, bool fp8, uint32_t ns, bool cpu_index) {
        constexpr uint32_t P = 4, B = 32, C = 64, D = 4, L = 2, Seed = 13;
        QwenExecutor e(model, weights, backend, P * B);
        KvMemStoreConfig cfg; cfg.block_tokens = B; cfg.n_subblocks = ns;
        cfg.index_placement = cpu_index ? KvMemIndexPlacement::CPU : KvMemIndexPlacement::GPU;
        cfg.index_staging_bytes = 2 * ns * D * L * (fp8 ? 2 : 4);
        e.block_store_ = std::make_unique<KvMemStore>(cfg);
        check(!e.block_store_->has_repository(), "tail probe must use legacy contiguous history");
        e.kvmem_enabled_ = true;
        e.kvmem_raw_k_block_major_ = true;
        e.kvmem_raw_k_row_bytes_ = D * (fp8 ? 1 : 4);
        e.kvmem_raw_layers_ = {0,1}; e.kvmem_raw_k_chunk_tokens_ = C;
        e.kvmem_raw_k_chunk_bytes_ = C * L * e.kvmem_raw_k_row_bytes_;
        e.kvmem_raw_k_valid_tokens_.assign(P * B, 1);
        e.kvmem_raw_k_chunks_.resize(2); e.kvmem_raw_k_dirty_.resize(2);
        for (auto &chunk : e.kvmem_raw_k_chunks_)
            chunk = std::make_unique<uint8_t[]>(e.kvmem_raw_k_chunk_bytes_);
        e.kvmem_qc_num_layers_ = L; e.kvmem_qc_n_subblocks_ = ns;
        e.kvmem_qc_layer_stride_blocks_ = P; e.kvmem_qc_total_blocks_ = 1;
        e.kvmem_qc_prompt_tokens_ = Seed; e.kvmem_qc_capture_active_ = true;
        if (cpu_index) e.kvmem_prepare_cpu_mean_index(P * ns * L * D, L, P, false);
        else e.g_kbar_multi_ = e.kvmem_alloc_mean_index_tensor(P * ns * L * D, "legacy_tail_index");
        const auto value = [](uint32_t row, uint32_t layer, uint32_t col) {
            return float(1u << ((row + 2 * layer + col) % 4));
        };
        auto seed = backend.tensor_f32(Seed * D, "tail_prefill_seed");
        for (uint32_t layer = 0; layer < L; ++layer) {
            std::vector<uint8_t> bytes(P * B * e.kvmem_raw_k_row_bytes_);
            std::vector<float> prefix(Seed * D);
            for (uint32_t row = 0; row < P * B; ++row) for (uint32_t col = 0; col < D; ++col) {
                const auto exponent = (row + 2 * layer + col) % 4;
                if (fp8) bytes[row * D + col] = uint8_t(0x38 + exponent * 8);
                else { const float v = value(row, layer, col); std::memcpy(bytes.data() + (row * D + col) * 4, &v, 4); }
                if (row < Seed) prefix[row * D + col] = value(row, layer, col);
            }
            e.kvmem_write_raw_k(layer, 0, bytes.data(), P * B, false);
            ok(backend.copy_bytes_from_host(*seed, 0, prefix.data(), prefix.size() * sizeof(float)));
            // Exercise the real prefill capture path first; the later calls
            // must merge generated raw K into these partial sub-block means.
            e.kvmem_index_rebuild_source_ = seed.get();
            e.kvmem_capture_kbar_multi(layer, Seed, 0, 0, D);
        }
        e.kvmem_index_rebuild_source_ = nullptr;
        check(e.kvmem_content_index_covers(Seed), "prefill seed index was not published");
        auto packed = e.kvmem_alloc_mean_index_tensor(ns * D, "tail_probe_read");
        auto widened = backend.tensor_f32(ns * D, "tail_probe_read_f32");
        const auto compare = [&](uint32_t end) {
            for (uint32_t layer = 0; layer < L; ++layer) for (uint32_t block = 0; block * B < end; ++block) {
                const uint64_t offset = uint64_t(block) * ns * D * packed->elem_size;
                if (cpu_index) {
                    ok(backend.copy_bytes_from_host(*packed, 0,
                        e.kvmem_mean_index_host_layer_data(layer) + offset, ns * D * packed->elem_size));
                } else {
                    ok(backend.copy_d2d(*packed, *e.g_kbar_multi_,
                        (uint64_t(layer) * P + block) * ns * D, ns * D));
                }
                ok(backend.widen_raw_k_to_f32_device(*packed, *widened, ns, D));
                std::vector<float> actual(ns * D);
                ok(backend.copy_to_host(*widened, actual.data(), 0, actual.size()));
                for (uint32_t sub = 0; sub < ns; ++sub) for (uint32_t col = 0; col < D; ++col) {
                    const uint32_t lo = block * B + sub * (B / ns);
                    const uint32_t hi = std::min(end, lo + B / ns);
                    double expected = 0;
                    for (uint32_t row = lo; row < hi; ++row) expected += value(row, layer, col);
                    if (hi > lo) expected /= hi - lo;
                    if (!std::isfinite(actual[sub * D + col]) ||
                        std::abs(actual[sub * D + col] - expected) >= (fp8 ? 0.005 : 1e-5)) {
                        throw std::runtime_error("raw-K decode tail mean mismatch: end=" +
                            std::to_string(end) + " layer=" + std::to_string(layer) +
                            " block=" + std::to_string(block) + " sub=" + std::to_string(sub) +
                            " col=" + std::to_string(col) + " actual=" + std::to_string(actual[sub * D + col]) +
                            " expected=" + std::to_string(expected));
                    }
                }
            }
        };
        compare(Seed);
        e.kvmem_raw_k_valid_tokens_[20] = 0;
        check(!e.kvmem_finalize_content_index_from_raw_k(29), "missing raw row was accepted");
        check(e.kvmem_qc_captured_tokens_ == Seed, "failed extension changed the published cursor");
        compare(Seed);
        e.kvmem_raw_k_valid_tokens_[20] = 1;
        for (uint32_t end : {29u, 37u, 64u, 70u}) {
            check(e.kvmem_finalize_content_index_from_raw_k(end), "legacy raw-K decode tail finalization failed");
            check(e.kvmem_content_index_covers(end) && e.kvmem_content_index_resume_compatible(end),
                  "complete decode tail is not available for warm checkpoint admission");
            compare(end);
        }
        std::cout << "legacy tail raw=" << (fp8 ? "FP8" : "FP32") << " ns=" << ns
                  << " index=" << (cpu_index ? "CPU" : "GPU")
                  << ": missing-row refusal and four unaligned extensions matched full-prefix means\n";
    }

    static void run(const QwenNativeModel &model, const QwenWeights &weights, DeviceBackend &backend, bool fp8, uint32_t ns) {
        constexpr uint32_t P = 4, B = 32, C = 64, D = 4, L = 2;
        QwenExecutor e(model, weights, backend, P * B);
        KvMemStoreConfig cfg; cfg.block_tokens = B; cfg.sink_blocks = 0;
        cfg.n_subblocks = ns; cfg.repository_blocks = P; cfg.index_placement = KvMemIndexPlacement::CPU;
        cfg.index_staging_bytes = 2 * ns * D * L * (fp8 ? 2 : 4); // two blocks per tile
        auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{P, B});
        e.block_store_ = std::make_unique<KvMemStore>(cfg, repo);
        e.kvmem_enabled_ = true;
        e.kvmem_repository_storage_generation_.resize(P); e.kvmem_repository_index_.resize(P);
        e.kv_pages_.configure_bounded(P * B, nullptr); e.mtp_kv_pages_.configure_bounded(P * B, nullptr);
        e.kvmem_raw_k_block_major_ = true;
        e.kvmem_raw_k_row_bytes_ = D * (fp8 ? 1 : 4);
        e.kvmem_raw_layers_ = {0, 1};
        e.kvmem_raw_k_chunk_tokens_ = C;
        e.kvmem_raw_k_chunk_bytes_ = C * L * e.kvmem_raw_k_row_bytes_;
        e.kvmem_raw_k_valid_tokens_.resize(P * B);
        e.kvmem_raw_k_chunks_.resize(2);
        e.kvmem_raw_k_dirty_.resize(2);
        for (auto &chunk : e.kvmem_raw_k_chunks_)
            chunk = std::make_unique<uint8_t[]>(e.kvmem_raw_k_chunk_bytes_);
        e.kvmem_qc_num_layers_ = L; e.kvmem_qc_n_subblocks_ = ns;
        e.kvmem_qc_layer_stride_blocks_ = P; e.kvmem_qc_total_blocks_ = P;
        e.kvmem_qc_capture_active_ = true;
        e.kvmem_prepare_cpu_mean_index(P * ns * L * D, L, P, false);
        check(e.g_kbar_stream_tile_blocks_ == 2, "repair staging is not fixed at two blocks");
        auto index_dev = e.kvmem_alloc_mean_index_tensor(ns * D, "test_index");
        auto widened = backend.tensor_f32(ns * D, "test_index_f32");
        auto source = backend.tensor_f32(B * D, "test_merge_source");
        const auto value = [](uint32_t row, uint32_t layer, uint32_t col, bool replacement) {
            return float(1u << ((row + 2 * layer + col + (replacement ? 2 : 0)) % 4));
        };
        const auto write = [&](const KvMemStorageSpan &span, uint32_t source_row, bool replacement) {
            for (uint32_t layer = 0; layer < L; ++layer) {
                std::vector<uint8_t> bytes(span.token_count * e.kvmem_raw_k_row_bytes_);
                for (uint32_t row = 0; row < span.token_count; ++row) for (uint32_t col = 0; col < D; ++col) {
                    const auto exponent = (row + source_row + 2 * layer + col + (replacement ? 2 : 0)) % 4;
                    if (fp8) bytes[row * D + col] = uint8_t(0x38 + exponent * 8); // exact e4m3 powers of two
                    else { const float v = float(1u << exponent); std::memcpy(bytes.data() + (row * D + col) * 4, &v, 4); }
                }
                e.kvmem_write_raw_k(layer, span.storage_begin, bytes.data(), span.token_count, false);
            }
            std::fill_n(e.kvmem_raw_k_valid_tokens_.begin() + span.storage_begin, span.token_count, 1);
        };
        const auto compare = [&](uint32_t physical, uint32_t n, uint32_t preserved) {
            for (uint32_t layer = 0; layer < L; ++layer) {
                ok(backend.copy_bytes_from_host(*index_dev, 0,
                    e.kvmem_mean_index_host_layer_data(layer) + physical * ns * D * index_dev->elem_size,
                    ns * D * index_dev->elem_size));
                ok(backend.widen_raw_k_to_f32_device(*index_dev, *widened, ns, D));
                std::vector<float> actual(ns * D); ok(backend.copy_to_host(*widened, actual.data(), 0, ns * D));
                for (uint32_t sub = 0; sub < ns; ++sub) for (uint32_t col = 0; col < D; ++col) {
                    double expected = 0;
                    const uint32_t lo = sub * (B / ns), hi = std::min(n, lo + B / ns);
                    for (uint32_t row = lo; row < hi; ++row) expected += value(row, layer, col, row >= preserved);
                    expected = hi > lo ? expected / (hi - lo) : 0;
                    check(std::abs(actual[sub * D + col] - expected) < (fp8 ? 0.005 : 1e-5),
                          "repaired/merged Mean-K differs from the retained raw prefix and new branch");
                }
            }
        };
        for (uint32_t round = 0; round < 64; ++round) {
            const uint32_t begin = static_cast<uint32_t>(e.block_store_->history_end());
            e.kvmem_prepare_repository_range(begin, 70, 0);
            const auto spans = e.kvmem_repository_append_->spans();
            for (const auto &span : spans) write(span, 0, false);
            e.kvmem_register_until(begin + 70);
            check(e.kvmem_finalize_content_index_from_raw_k(begin + 70), "bounded index finalization failed");
            compare(spans[0].handle.slot, 32, 32);
            compare(spans[2].handle.slot, 6, 6);
            check(!e.kvmem_content_index_covers(begin + 37) &&
                  e.kvmem_content_index_resume_compatible(begin + 37),
                  "repairable checkpoint tail was scored as exact or rejected for warm admission");
            const auto prefix_row = spans[1].storage_begin + 4;
            e.kvmem_raw_k_valid_tokens_[prefix_row] = 0;
            check(!e.kvmem_content_index_resume_compatible(begin + 37),
                  "warm partial-tail admission accepted incomplete retained raw K");
            e.kvmem_raw_k_valid_tokens_[prefix_row] = 1;
            if (round == 0) {
                e.kvmem_clear_cpu_mean_index_storage();
                e.kvmem_prepare_cpu_mean_index(P * ns * L * D, L, P, false);
                check(!e.kvmem_content_index_covers(begin + 70),
                      "reallocated Mean-K authority retained stale content certificates");
                check(e.kvmem_finalize_content_index_from_raw_k(begin + 70),
                      "raw K could not restore the reallocated bounded authority");
                compare(spans[0].handle.slot, 32, 32);
                compare(spans[2].handle.slot, 6, 6);
            }

            // Verify cold checkpoint metadata survives exact GPU/CPU storage.
            QwenExecutor::StateSnapshot src, dst;
            src.repository_checkpoint = e.block_store_->repository_checkpoint(begin + 70);
            detail::PrefixStateStorage::copy_metadata(dst, src);
            check(dst.repository_checkpoint == src.repository_checkpoint, "prefix storage lost the repository lease");
            e.kvmem_truncate_to(begin + 37);
            check(!dst.repository_checkpoint->valid(), "discarded checkpoint was resurrected through CPU metadata");
            src.repository_checkpoint.reset(); dst.repository_checkpoint.reset();
            check(!e.kvmem_content_index_covers(begin + 37), "rewind still advertised a stale partial mean");
            check(repo->try_reclaim(spans[2].handle, [](const auto &) { return true; }) ==
                  KvMemRepository::ReclaimResult::Reclaimed, "replay suffix retained a dead checkpoint lease");

            e.kvmem_registered_pos_ = begin + 37;
            e.kvmem_prepare_repository_range(begin + 37, 27, 37);
            e.kvmem_qc_captured_tokens_ = begin + 37;
            const uint32_t tile = e.kvmem_begin_mean_index_capture((begin + B) / B, 1, true, begin + 64);
            compare(spans[1].handle.slot, 5, 5); // only the surviving prefix seeded this merge
            for (uint32_t layer = 0; layer < L; ++layer) {
                std::vector<float> rows(27 * D);
                for (uint32_t row = 0; row < 27; ++row) for (uint32_t col = 0; col < D; ++col)
                    rows[row * D + col] = value(row + 5, layer, col, true);
                ok(backend.copy_bytes_from_host(*source, 0, rows.data(), rows.size() * sizeof(float)));
                ok(backend.block_kmean_content_batch_merge_device(*source, *e.g_kbar_stream_slots_[tile].device,
                    layer, 1, D, 27, B, 5, 1, D, 0, 0, model.config().rope_theta, ns));
            }
            write(e.kvmem_repository_append_->spans().front(), 5, true);
            e.kvmem_finish_mean_index_capture(tile, (begin + B) / B, 1);
            e.kvmem_drain_mean_index_capture();
            e.kvmem_register_until(begin + 64);
            compare(spans[1].handle.slot, 32, 5);
            check(e.kvmem_finalize_content_index_from_raw_k(begin + 64), "new branch index cannot be published");
            for (uint32_t i = 0; i < 2; ++i) {
                const auto h = spans[i].handle;
                e.block_store_->set_block_tier(h.slot, KvTier::CPU, h.slot, -1, h.generation);
                check(repo->retire_if_unpinned(h), "index repair retained a payload lease");
                check(repo->try_reclaim(h, [](const auto &) { return true; }) == KvMemRepository::ReclaimResult::Reclaimed,
                      "index payload did not become reusable");
            }
            check(repo->ready_slots() == P, "index repair leaked physical slots");
        }
        check(e.block_store_->history_end() == 4096 && e.block_store_->block_count() == P,
              "index repair grew the physical directory with historical length");
        check(e.kvmem_index_rebuild_raw_dev_->count <= 2 * B * D, "raw repair scratch grew past its fixed tile");
        std::cout << (fp8 ? "FP8" : "FP32") << " ns=" << ns << " raw / CPU Mean-K: 64 rewind/append cycles passed; history=4096, capacity=128\n";
    }
};
}

int main() {
    const auto dir = std::filesystem::temp_directory_path() / ("qw3-index-test-" + std::to_string(getpid()));
    try {
        std::filesystem::create_directories(dir);
        std::ofstream(dir / "config.json") << R"({"text_config":{"model_type":"qwen3_5_text","num_hidden_layers":2,
          "hidden_size":4,"intermediate_size":4,"max_position_embeddings":128,"num_attention_heads":1,
          "num_key_value_heads":1,"head_dim":4,"linear_conv_kernel_dim":4,"linear_num_key_heads":1,
          "linear_num_value_heads":1,"linear_key_head_dim":4,"linear_value_head_dim":4,"full_attention_interval":1,"vocab_size":4}})";
        setenv("QW3_PAGED_KV_PAGE_SIZE", "16", 1);
        auto backend = make_cuda_device_backend(LinearBackend::Auto);
        check(bool(backend), "CUDA backend unavailable"); ok(backend->begin());
        {
            QwenNativeModel model(std::make_unique<IndexSource>(dir.string()));
            QwenWeights weights(model, *backend, false, false);
            for (bool fp8 : {false, true}) {
                setenv("QW3_KV_DTYPE", fp8 ? "fp8" : "fp32", 1);
                for (uint32_t ns : {1u, 4u}) KvPageTableTestAccess::run(model, weights, *backend, fp8, ns);
                KvPageTableTestAccess::query_tiles(model, weights, *backend, fp8);
                for (uint32_t ns : {1u, 4u}) for (bool cpu_index : {false, true})
                    KvPageTableTestAccess::legacy_tail(model, weights, *backend, fp8, ns, cpu_index);
            }
        }
        ok(backend->end()); std::filesystem::remove_all(dir);
        return 0;
    } catch (const std::exception &e) {
        std::filesystem::remove_all(dir); std::cerr << e.what() << '\n'; return 1;
    }
}
