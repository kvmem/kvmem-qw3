#include "qwen_executor.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>

namespace qw3 {
struct KvPageTableTestAccess {
    static void run(const QwenNativeModel &model, const QwenWeights &weights, DeviceBackend &backend) {
        const auto require = [](bool ok, const char *what) { if (!ok) throw std::runtime_error(what); };
        const auto bytes = [&](const DeviceTensor &t) {
            std::vector<uint8_t> out(t.count * t.elem_size);
            require(backend.copy_bytes_to_host(t, out.data(), 0, out.size()).ok, "read failed");
            return out;
        };
        QwenExecutor e(model, weights, backend, 4096);
        KvMemStoreConfig cfg;
        cfg.block_tokens = 32; cfg.select_budget = cfg.prefill_budget = 1024;
        cfg.gen_budget = 1024; cfg.sink_blocks = 1; cfg.recent_blocks = 1;
        cfg.immutable_source_k = true;
        cfg.cpu_tier_bytes = 1ull << 30;
        cfg.gpu_memory_ratio = 1;
        if (std::getenv("QW3_TEST_BOUNDED")) {
            cfg.repository_blocks = 128;
            cfg.index_placement = KvMemIndexPlacement::CPU;
            cfg.select_budget = cfg.prefill_budget = 256;
            cfg.gen_budget = 128;
        }
        e.configure_kvmem(cfg); e.set_kvmem_enabled(true);
        e.kvmem_set_query_span(496, 512, 512);
        std::vector<uint32_t> tokens(512);
        for (uint32_t i = 0; i < tokens.size(); ++i) tokens[i] = 100 + i % 300;
        require(e.forward_n_tokens(tokens, false).ok, "history forward failed");
        e.kvmem_register_append(tokens.size());
        for (uint32_t begin = 0; begin < tokens.size(); begin += cfg.block_tokens) {
            const auto found = e.kvmem_blend_tokens_.find(begin);
            require(found != e.kvmem_blend_tokens_.end() && found->second.size() == cfg.block_tokens,
                    "pressure admission pruned live source tokens");
            require(std::equal(found->second.begin(), found->second.end(), tokens.begin() + begin),
                    "retained source tokens do not match history");
        }
        auto boundary = e.snapshot_state();
        e.kvmem_set_selection({0, 4, 5, 9, 10, 14});
        e.kvmem_begin_query_replay(boundary, {0, 4, 5, 9, 10, 14}, false, true);
        std::vector<std::vector<uint8_t>> original, recurrent;
        for (auto il : e.kvmem_raw_layers_) {
            original.push_back(bytes(e.k_cache(il))); original.push_back(bytes(e.v_cache(il)));
        }
        for (const auto &s : e.recurrent_states_) if (s) recurrent.push_back(bytes(*s));
        for (const auto &s : e.conv_states_) if (s) recurrent.push_back(bytes(*s));
        const auto h = bytes(*e.h_);
        const auto pages = e.window_pages_host_;
        const auto pos = e.position_;
        const auto window = e.window_query_pos_;
        const auto raw_bytes = e.kvmem_raw_layers_.size() * 32 * e.kvmem_raw_k_row_bytes_;
        std::vector<uint8_t> raw_before(raw_bytes), raw_after(raw_bytes);
        e.kvmem_read_raw_k_block(128, 32, raw_before.data(), false);
        e.kvmem_blend_selected();
        require(!e.kvmem_blend_backup_.empty(), "blend did not execute");
        require(e.position_ == pos && e.window_query_pos_ == window && e.window_pages_host_ == pages,
                "blend changed coordinates/pages");
        require(bytes(*e.h_) == h, "blend changed boundary hidden state");
        size_t at = 0;
        for (const auto &s : e.recurrent_states_) if (s) require(bytes(*s) == recurrent.at(at++), "recurrent changed");
        for (const auto &s : e.conv_states_) if (s) require(bytes(*s) == recurrent.at(at++), "conv changed");
        bool changed = false;
        at = 0;
        for (auto il : e.kvmem_raw_layers_) {
            for (auto *t : {&e.k_cache(il), &e.v_cache(il)}) {
                const auto after = bytes(*t);
                const auto &before = original.at(at++);
                // Only backed-up physical rows may differ; all other tokens,
                // including selected-but-unrefreshed blocks, must be bit exact.
                std::vector<bool> allowed(after.size(), false);
                for (const auto &saved : e.kvmem_blend_backup_) if (saved.layer == il) {
                    for (const auto &piece : saved.pieces) {
                        const uint64_t begin = piece.offset * t->elem_size;
                        std::fill(allowed.begin() + begin, allowed.begin() + begin + piece.count * t->elem_size, true);
                    }
                }
                for (size_t i = 0; i < after.size(); ++i) if (after[i] != before[i]) {
                    require(allowed[i], "unselected KV row changed"); changed = true;
                }
            }
        }
        const char *alpha_env = std::getenv("QW3_KVMEM_BLEND_ALPHA");
        require(changed == (!alpha_env || std::strtod(alpha_env, nullptr) != 0),
                "selected KV change does not match alpha endpoint");
        QwenExecutor::StateSnapshot during_blend;
        e.capture_state(during_blend);
        require(!e.kvmem_blend_backup_.empty(), "checkpoint prematurely removed the blend window");
        e.kvmem_read_raw_k_block(128, 32, raw_after.data(), false);
        require(raw_before == raw_after, "immutable raw K changed");
        // Changing future KV must not affect the recomputed historical rows.
        std::vector<std::vector<uint8_t>> expected;
        std::ofstream before_dump, after_dump;
        if (const char *prefix = std::getenv("QW3_TEST_BLEND_DUMP")) {
            before_dump.open(std::string(prefix) + ".old", std::ios::binary);
            after_dump.open(std::string(prefix) + ".new", std::ios::binary);
            require(bool(before_dump) && bool(after_dump), "cannot open blend dump");
        }
        for (const auto &saved : e.kvmem_blend_backup_) {
            for (auto *cache : {&e.k_cache(saved.layer), &e.v_cache(saved.layer)}) {
                const auto full = bytes(*cache);
                std::vector<uint8_t> rows;
                for (const auto &piece : saved.pieces)
                    rows.insert(rows.end(), full.begin() + piece.offset * cache->elem_size,
                                full.begin() + (piece.offset + piece.count) * cache->elem_size);
                if (before_dump.is_open()) {
                    const auto packed = bytes(cache == &e.k_cache(saved.layer) ? *saved.k : *saved.v);
                    before_dump.write(reinterpret_cast<const char *>(packed.data()), packed.size());
                    after_dump.write(reinterpret_cast<const char *>(rows.data()), rows.size());
                    require(bool(before_dump) && bool(after_dump), "cannot write blend dump");
                }
                expected.push_back(std::move(rows));
            }
        }
        e.kvmem_blend_restore();
        struct Future { DeviceTensor *target; uint64_t offset; std::unique_ptr<DeviceTensor> copy; };
        std::vector<Future> future;
        const auto per_pos = model.config().n_kv_heads * model.config().head_dim;
        const uint64_t future_offset = uint64_t(e.window_pages_host_.back()) * e.kv_pages_.page_size * per_pos;
        const uint64_t future_count = uint64_t(e.kv_pages_.page_size) * per_pos;
        for (auto il : e.kvmem_raw_layers_) for (auto *cache : {&e.k_cache(il), &e.v_cache(il)}) {
            auto copy = cache->dtype == DeviceTensorDType::FP8_E4M3
                ? backend.tensor_fp8_kv(future_count, "future") : backend.tensor_f16(future_count, "future");
            require(backend.copy_d2d(*copy, *cache, future_offset, future_count).ok, "future backup failed");
            const std::vector<uint8_t> zeros(future_count * cache->elem_size, 0);
            require(backend.copy_bytes_from_host(*cache, future_offset * cache->elem_size,
                                                 zeros.data(), zeros.size()).ok, "future mutation failed");
            future.push_back({cache, future_offset, std::move(copy)});
        }
        e.kvmem_blend_selected();
        at = 0;
        for (const auto &saved : e.kvmem_blend_backup_) {
            for (auto *cache : {&e.k_cache(saved.layer), &e.v_cache(saved.layer)}) {
                const auto full = bytes(*cache);
                std::vector<uint8_t> rows;
                for (const auto &piece : saved.pieces)
                    rows.insert(rows.end(), full.begin() + piece.offset * cache->elem_size,
                                full.begin() + (piece.offset + piece.count) * cache->elem_size);
                require(rows == expected.at(at++), "future KV leaked into historical recompute");
            }
        }
        e.kvmem_blend_restore();
        for (const auto &saved : future)
            require(backend.copy_d2d_into(*saved.target, saved.offset, *saved.copy, 0, saved.copy->count).ok,
                    "future restore failed");
        at = 0;
        for (auto il : e.kvmem_raw_layers_) {
            require(bytes(e.k_cache(il)) == original.at(at++), "K restore not bit exact");
            require(bytes(e.v_cache(il)) == original.at(at++), "V restore not bit exact");
        }
        require(e.forward_n_tokens({100, 101, 102, 103}, true).ok, "query replay failed");
        e.kvmem_end_query_replay();
        std::cout << "blend model: selected KV follows alpha; other rows, source restore, immutable raw K, causal mask, recurrent/conv, hidden, pages and coordinates passed\n";
    }
};
}
int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: qw3-kvmem-blend-model-probe MODEL");
        setenv("QW3_KVMEM_BLEND_RATIO", "0.5", 1);
        setenv("QW3_KVMEM_BLEND_MAX_TOKENS", "64", 1);
        setenv("QW3_PAGED_KV_PAGE_SIZE", "16", 1);
        auto backend = qw3::make_cuda_device_backend(qw3::LinearBackend::Auto);
        if (!backend || !backend->begin().ok) throw std::runtime_error("CUDA backend unavailable");
        qw3::QwenNativeModel model(qw3::open_model_source(argv[1]));
        qw3::QwenWeights weights(model, *backend, false, false);
        qw3::KvPageTableTestAccess::run(model, weights, *backend);
        if (!backend->end().ok) throw std::runtime_error("backend end failed");
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
