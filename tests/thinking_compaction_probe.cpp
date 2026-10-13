#include "qwen_executor.hpp"
#include <iostream>
#include <stdexcept>

// Manual CUDA oracle: discarded reasoning must have no effect on the state of
// prefix+replacement. Uses synthetic tokens, not a task-quality measurement.
namespace qw3 {
struct KvPageTableTestAccess {
    using Bytes = std::vector<std::pair<std::string, std::vector<uint8_t>>>;
    static void tensor(Bytes &out, DeviceBackend &device, const std::string &name,
                       const DeviceTensor *value) {
        if (!value) return;
        auto &bytes = out.emplace_back(name, std::vector<uint8_t>(value->count * value->elem_size)).second;
        const auto status = device.copy_bytes_to_host(*value, bytes.data(), 0, bytes.size());
        if (!status.ok) throw std::runtime_error(status.message);
    }
    static Bytes states(QwenExecutor &e) {
        Bytes out;
        tensor(out, e.backend_, "hidden", e.h_.get());
        tensor(out, e.backend_, "mtp-hidden", e.mtp_prefix_h_.get());
        for (size_t i = 0; i < e.recurrent_states_.size(); ++i) {
            tensor(out, e.backend_, "recurrent-" + std::to_string(i), e.recurrent_states_[i].get());
            tensor(out, e.backend_, "conv-" + std::to_string(i), e.conv_states_[i].get());
        }
        if (auto *draft = e.dflash_executor()) {
            const auto snapshot = draft->snapshot();
            out.emplace_back("draft-position=" + std::to_string(snapshot.next_position) +
                " rows=" + std::to_string(snapshot.rows) + " rope=" + std::to_string(snapshot.rope_next),
                std::vector<uint8_t>{});
            for (size_t i = 0; i < snapshot.raw_keys.size(); ++i) {
                tensor(out, e.backend_, "draft-K-" + std::to_string(i), snapshot.raw_keys[i].get());
                tensor(out, e.backend_, "draft-V-" + std::to_string(i), snapshot.values[i].get());
            }
        }
        return out;
    }
    static void equal(const Bytes &a, const Bytes &b, const char *where) {
        if (a.size() != b.size()) throw std::runtime_error(std::string(where) + " shape changed");
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i]) throw std::runtime_error(std::string(where) + " differs: " + a[i].first);
    }
    static Bytes query(QwenExecutor &e) {
        e.kvmem_drain_query_capture();
        Bytes out;
        const uint64_t row_bytes = uint64_t(e.model_.config().n_heads) *
            e.model_.config().head_dim * e.g_query_multi_->elem_size;
        for (uint32_t layer = 0; layer < e.kvmem_qc_num_layers_; ++layer) {
            auto &bytes = out.emplace_back("query-" + std::to_string(layer),
                std::vector<uint8_t>(row_bytes * e.g_query_multi_count_)).second;
            const uint64_t offset = layer * row_bytes * e.kvmem_query_span_;
            if (e.kvmem_query_host_capture_) {
                const auto *host = reinterpret_cast<const uint8_t *>(e.kvmem_query_multi_host_data());
                std::copy(host + offset, host + offset + bytes.size(), bytes.begin());
            } else {
                const auto status = e.backend_.copy_bytes_to_host(*e.g_query_multi_,
                    bytes.data(), offset, bytes.size());
                if (!status.ok) throw std::runtime_error(status.message);
            }
        }
        return out;
    }
    static Bytes run(const QwenNativeModel &model, const QwenWeights &weights,
                     DeviceBackend &device, uint32_t boundary, unsigned branch,
                     unsigned query_kind = 0, const DFlashWeights *draft_weights = nullptr) {
        QwenExecutor e(model, weights, device, 8192);
        auto *draft = draft_weights ? &e.enable_dflash(*draft_weights) : nullptr;
        const auto draft_position = [&]() { return draft ? draft->next_position() : e.mtp_prefix_len_; };
        KvMemStoreConfig cfg;
        cfg.block_tokens = 32; cfg.repository_blocks = 256;
        cfg.select_budget = cfg.prefill_budget = 256; cfg.gen_budget = 1024;
        cfg.sink_blocks = 1; cfg.recent_blocks = 4;
        cfg.immutable_source_k = true; cfg.index_placement = KvMemIndexPlacement::CPU;
        cfg.mtp_enabled = !draft; cfg.cpu_tier_bytes = 1ull << 30;
        cfg.gpu_memory_ratio = 0; cfg.estimated_gpu_block_capacity = 128;
        e.configure_kvmem(cfg);
        e.set_kvmem_enabled(true);
        e.kvmem_gc_begin_step();
        e.kvmem_set_query_score_budget_hint(query_kind == 2 ? 16 : 256);
        e.kvmem_set_query_span(query_kind ? boundary - 40 : 0,
            query_kind ? boundary - 8 : 0, boundary, 0, false, true);
        const auto forward = [&](uint32_t count, uint32_t token, bool publish) {
            std::vector<uint32_t> ids(count, token);
            const auto base = e.position();
            e.kvmem_set_defer_prefill_pressure(true);
            auto result = e.forward_n_tokens(ids, true);
            if (!result.ok) throw std::runtime_error("target forward failed");
            if (draft) {
                const auto features = e.dflash_features();
                if (!features.data || features.rows != count || features.logical_base != base)
                    throw std::runtime_error("DFlash context metadata missing");
                draft->append_context(*features.data, count, base);
            } else {
                const auto mtp = e.prime_mtp_prefix_from_last_batch_at(ids, base, e.last_forward_rope_base());
                if (!mtp.ok) throw std::runtime_error("MTP prefix failed");
            }
            e.kvmem_set_defer_prefill_pressure(false);
            if (publish) e.kvmem_register_append(count);
            return result;
        };
        for (uint32_t offset = 0; offset < boundary;) {
            const auto count = std::min(32u, boundary - offset);
            forward(count, 100 + offset % 17, true); offset += count;
        }
        // The short case must also cover the initial dense logical window,
        // before any explicit selection. Private-tail cancellation recycles
        // storage slots while the logical prefix stays fixed.
        if (boundary > cfg.select_budget) e.kvmem_reselect_prefill_pressure();
        if (query_kind) {
            if (!e.kvmem_finalize_content_index_from_raw_k(boundary) ||
                !e.kvmem_query_capture_complete() || !e.kvmem_query_selection_ready())
                throw std::runtime_error("prefix query not captured");
            if (query_kind >= 3) {
                QwenExecutor::StateSnapshot guided_boundary;
                e.capture_transient_state(guided_boundary);
                if (!e.kvmem_begin_guided_query_probe(64))
                    throw std::runtime_error("guided Q admission failed");
                e.kvmem_guided_query_capture_this_token_ = true;
                const auto count = query_kind == 3 ? 7u : 48u;
                forward(count, 600, false);
                if (e.kvmem_end_guided_query_probe() != count)
                    throw std::runtime_error("guided Q capture failed");
                e.restore_state(guided_boundary);
            }
        }
        const auto original_query = query_kind ? query(e) : Bytes{};
        const auto original_query_count = e.kvmem_query_captured_tokens();
        const auto original_query_complete = e.kvmem_query_capture_complete();
        const auto original = states(e);
        const auto history = e.block_store_->history_end();
        const auto generation = e.kvmem_selection_generation();
        if (branch == 1) {
            if (!e.kvmem_begin_private_generation_probe(512))
                throw std::runtime_error("private branch admission failed");
            QwenExecutor::StateSnapshot checkpoint;
            e.capture_transient_state(checkpoint);
            for (uint32_t i = 0; i < 4; ++i) {
                forward(32, 700 + i, false);
                // Full-snapshot MTP rejection is nested in the outer private
                // transaction; it must retain all earlier private KV pages.
                QwenExecutor::StateSnapshot nested;
                e.capture_state(nested);
                const auto nested_state = states(e);
                const auto append_serial = e.kvmem_repository_append_serial_;
                forward(16, 800 + i, false);
                e.restore_state(nested);
                if (!e.kvmem_repository_append_ ||
                    append_serial != e.kvmem_repository_append_serial_)
                    throw std::runtime_error("nested rejection discarded the private append");
                equal(nested_state, states(e), "nested rejection");
            }
            // The summarizer uses main-model decode without committing its MTP
            // or content index, just like the production private branch.
            if (!e.forward_n_tokens(std::vector<uint32_t>(64, 900), true).ok)
                throw std::runtime_error("summary branch failed");
            if (e.block_store_->history_end() != history ||
                e.kvmem_selection_generation() != generation)
                throw std::runtime_error("private branch mutated committed history/selection");
            e.kvmem_cancel_guided_query_probe();
            e.restore_state(checkpoint);
            if (e.position() != boundary || draft_position() != boundary)
                throw std::runtime_error("rollback position mismatch");
            equal(original, states(e), "rollback");
        }
        if (branch == 2) {
            QwenExecutor::StateSnapshot checkpoint;
            e.capture_transient_state(checkpoint);
            // Production now publishes ordinary reasoning. Only the summary
            // is private; success rewinds the published tail as well.
            for (uint32_t i = 0; i < 4; ++i) forward(32, 700 + i, true);
            QwenExecutor::StateSnapshot original_end;
            e.capture_state(original_end);
            const auto end_state = states(e);
            const auto end_history = e.block_store_->history_end();
            const auto summarize = [&]() {
                if (e.kvmem_begin_bounded_private_generation_probe(256) < 64)
                    throw std::runtime_error("summary branch admission failed");
                if (!e.forward_n_tokens(std::vector<uint32_t>(64, 900), true).ok)
                    throw std::runtime_error("summary branch failed");
                if (e.block_store_->history_end() != end_history)
                    throw std::runtime_error("summary published private history");
                e.kvmem_cancel_guided_query_probe();
            };
            summarize();
            e.restore_state(original_end);
            equal(end_state, states(e), "invalid summary without replay");
            summarize();
            original_end = QwenExecutor::StateSnapshot{};
            // Reject invalid provenance before mutating the published history.
            bool rejected = false;
            try { e.restore_state_preserving_query(checkpoint, boundary + 1); }
            catch (const std::runtime_error &) { rejected = true; }
            if (!rejected || e.block_store_->history_end() != end_history)
                throw std::runtime_error("unsafe query preservation was accepted");
            e.restore_state_preserving_query(checkpoint, boundary);
            if (e.position() != boundary || draft_position() != boundary ||
                e.block_store_->history_end() != history)
                throw std::runtime_error("published reasoning rollback position mismatch");
            equal(original, states(e), "published reasoning rollback");
        }
        if (query_kind) {
            equal(original_query, query(e), "retained query");
            if (e.kvmem_query_begin_ != boundary - 40 ||
                e.kvmem_query_end_ != boundary - 8 ||
                e.kvmem_query_captured_tokens() != original_query_count ||
                e.kvmem_query_capture_complete() != original_query_complete ||
                !e.kvmem_query_selection_ready())
                throw std::runtime_error("query readiness lost after rollback");
        }
        const uint32_t end = ((boundary + 31) / 32 + 1) * 32;
        for (uint32_t offset = boundary; offset < end;) {
            const auto count = std::min(32u, end - offset);
            forward(count, 401, true); offset += count;
        }
        if (!e.kvmem_finalize_content_index_from_raw_k(end))
            throw std::runtime_error("replacement index finalization failed");
        auto out = states(e);
        tensor(out, device, "logits", e.logits_.get());
        const auto ids = e.block_store_->history_span_blocks(end - 32, end, true);
        if (ids.size() != 1) throw std::runtime_error("missing replacement block");
        const auto storage = e.block_store_->blocks()[ids[0]].orig_pos_start;
        for (const bool mtp : {false, true}) {
            if (mtp && draft) continue;
            const auto size = (mtp ? 1 : e.kvmem_raw_layers_.size()) * 32 * e.kvmem_raw_k_row_bytes_;
            auto &raw = out.emplace_back(mtp ? "mtp-raw-K" : "main-raw-K", std::vector<uint8_t>(size)).second;
            e.kvmem_read_raw_k_block(storage, 32, raw.data(), mtp);
        }
        const auto cache = [&](const std::string &name, DeviceTensor &value, const QwenExecutor::KvPageTable &pages) {
            const uint64_t page_bytes = uint64_t(pages.page_size) * model.config().n_kv_heads *
                model.config().head_dim * value.elem_size;
            auto &bytes = out.emplace_back(name, std::vector<uint8_t>(32 / pages.page_size * page_bytes)).second;
            for (uint32_t i = 0; i < 32 / pages.page_size; ++i) {
                const auto page = pages.pages.at(storage / pages.page_size + i);
                if (page < 0) throw std::runtime_error("replacement GPU page missing");
                const auto status = device.copy_bytes_to_host(value, bytes.data() + i * page_bytes,
                    page * page_bytes, page_bytes);
                if (!status.ok) throw std::runtime_error(status.message);
            }
        };
        for (auto layer : e.kvmem_raw_layers_) {
            cache("K-" + std::to_string(layer), e.k_cache(layer), e.kv_pages_);
            cache("V-" + std::to_string(layer), e.v_cache(layer), e.kv_pages_);
        }
        if (!draft) {
            cache("mtp-K", e.mtp_k_cache(), e.mtp_kv_pages_);
            cache("mtp-V", e.mtp_v_cache(), e.mtp_kv_pages_);
        }
        if (e.block_store_->history_end() != end || draft_position() != end)
            throw std::runtime_error("replacement commit mismatch");
        if (query_kind) {
            equal(original_query, query(e), "query after replacement");
            if (!e.kvmem_stash_query() || e.kvmem_stashed_query_tokens() != original_query_count)
                throw std::runtime_error("warm query snapshot unavailable");
            // Ordinary cross-request divergence must still invalidate Q.
            e.kvmem_truncate_to(end);
            if (e.kvmem_query_selection_ready() || e.kvmem_query_capture_complete() ||
                e.kvmem_query_expected_tokens() || e.kvmem_query_captured_tokens())
                throw std::runtime_error("general truncate retained a stale query");
        }
        e.kvmem_gc_end_step();
        return out;
    }
};
}

int main(int argc, char **argv) {
    try {
        if (argc < 2 || argc > 3) throw std::invalid_argument("usage: qw3-thinking-compaction-probe MODEL [DFLASH]");
        setenv("QW3_PAGED_KV_PAGE_SIZE", "16", 1);
        setenv("QW3_KV_DTYPE", "fp8", 1);
        auto device = qw3::make_cuda_device_backend(qw3::LinearBackend::Auto);
        if (!device) throw std::runtime_error("CUDA unavailable");
        auto status = device->begin();
        if (!status.ok) throw std::runtime_error(status.message);
        {
            qw3::QwenNativeModel model(qw3::open_model_source(argv[1]));
            qw3::QwenWeights weights(model, *device, false, argc == 2);
            std::unique_ptr<qw3::DFlashModel> draft_model;
            std::unique_ptr<qw3::DFlashWeights> draft;
            if (argc == 3) {
                draft_model = std::make_unique<qw3::DFlashModel>(argv[2]);
                draft = std::make_unique<qw3::DFlashWeights>(*draft_model, *device);
            }
            for (const uint32_t boundary : {63u, 511u, 1535u, 4095u}) {
                const auto reference = qw3::KvPageTableTestAccess::run(model, weights, *device, boundary, 0, 0, draft.get());
                for (unsigned branch : {1u, 2u}) {
                    const auto compacted = qw3::KvPageTableTestAccess::run(model, weights, *device, boundary, branch, 0, draft.get());
                    qw3::KvPageTableTestAccess::equal(reference, compacted, "replacement vs clean reference");
                }
                std::cout << "boundary=" << boundary << " recurrent=exact conv=exact main_kv=exact "
                          << (draft ? "dflash=exact" : "mtp=exact") << " logits=exact\n" << std::flush;
            }
            for (unsigned query_kind : {1u, 2u, 3u, 4u}) {
                const auto reference = qw3::KvPageTableTestAccess::run(model, weights, *device, 511, 0, query_kind, draft.get());
                const auto compacted = qw3::KvPageTableTestAccess::run(model, weights, *device, 511, 2, query_kind, draft.get());
                qw3::KvPageTableTestAccess::equal(reference, compacted, "query replacement vs clean reference");
                std::cout << "query_kind=" << query_kind << " query=exact state=exact warm_stash=ready unsafe_restore=rejected general_truncate=invalidated\n";
            }
        }
        status = device->end();
        if (!status.ok) throw std::runtime_error(status.message);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
