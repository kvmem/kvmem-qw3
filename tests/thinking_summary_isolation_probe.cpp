#include "qwen_executor.hpp"
#include "qw3/tokenizer.hpp"
#include "thinking_compaction.hpp"
#include <iostream>
#include <stdexcept>

// Manual GPU oracle: equal current-thinking input must produce identical logits
// under different parent histories. Child execution/destruction must not change
// the parent's KV, recurrent/conv, logits, repository or page accounting.
namespace qw3 {
struct KvPageTableTestAccess {
    using Bytes = std::vector<uint8_t>;
    static void require(bool value, const char *message) {
        if (!value) throw std::runtime_error(message);
    }
    static void append(Bytes &out, DeviceBackend &device, const DeviceTensor *tensor,
                       uint64_t offset = 0, uint64_t length = 0) {
        if (!tensor) return;
        if (!length) length = tensor->count * tensor->elem_size;
        const auto base = out.size(); out.resize(base + length);
        auto status = device.copy_bytes_to_host(*tensor, out.data() + base, offset, length);
        if (!status.ok) throw std::runtime_error(status.message);
    }
    static Bytes parent_state(QwenExecutor &parent) {
        Bytes bytes;
        append(bytes, parent.backend_, parent.h_.get());
        append(bytes, parent.backend_, parent.logits_.get());
        for (const auto &t : parent.recurrent_states_) append(bytes, parent.backend_, t.get());
        for (const auto &t : parent.conv_states_) append(bytes, parent.backend_, t.get());
        for (uint32_t layer = 0; layer < parent.model_.config().n_layers; ++layer) {
            if (!parent.model_.config().is_standard_attention_layer(layer)) continue;
            for (auto *tensor : {&parent.k_cache(layer), &parent.v_cache(layer)}) {
                const uint64_t page_bytes = uint64_t(parent.kv_pages_.page_size) *
                    parent.model_.config().n_kv_heads * parent.model_.config().head_dim * tensor->elem_size;
                for (auto page : parent.kv_pages_.pages)
                    if (page >= 0) append(bytes, parent.backend_, tensor, uint64_t(page) * page_bytes, page_bytes);
            }
        }
        if (auto *draft = parent.dflash_executor()) {
            const auto snapshot = draft->snapshot();
            for (uint64_t value : {uint64_t(snapshot.next_position), uint64_t(snapshot.rows),
                                   uint64_t(snapshot.rope_next)}) {
                const auto *raw = reinterpret_cast<const uint8_t *>(&value);
                bytes.insert(bytes.end(), raw, raw + sizeof(value));
            }
            for (size_t i = 0; i < snapshot.raw_keys.size(); ++i) {
                append(bytes, parent.backend_, snapshot.raw_keys[i].get());
                append(bytes, parent.backend_, snapshot.values[i].get());
            }
        }
        return bytes;
    }
    static Bytes run(const QwenNativeModel &model, const QwenWeights &weights,
                     DeviceBackend &device, const std::vector<uint32_t> &input, uint32_t parent_token,
                     const DFlashWeights *draft_weights) {
        QwenExecutor parent(model, weights, device, 8192);
        auto *draft = draft_weights ? &parent.enable_dflash(*draft_weights) : nullptr;
        KvMemStoreConfig cfg;
        cfg.block_tokens = 32; cfg.repository_blocks = 256;
        cfg.select_budget = cfg.prefill_budget = 256; cfg.gen_budget = 1024;
        cfg.sink_blocks = 1; cfg.recent_blocks = 4;
        cfg.immutable_source_k = true; cfg.index_placement = KvMemIndexPlacement::CPU;
        cfg.mtp_enabled = false; cfg.cpu_tier_bytes = 512ull << 20;
        cfg.gpu_memory_ratio = 0; cfg.estimated_gpu_block_capacity = 128;
        parent.configure_kvmem(cfg); parent.set_kvmem_enabled(true); parent.reset_state();
        parent.kvmem_gc_begin_step();
        const uint32_t history_tokens = parent_token == 100 ? 64 : 160;
        require(parent.forward_n_tokens(std::vector<uint32_t>(history_tokens, parent_token)).ok, "parent prefill failed");
        if (draft) {
            const auto features = parent.dflash_features();
            require(features.data && features.rows == history_tokens && features.logical_base == 0,
                    "parent DFlash context missing");
            draft->append_context(*features.data, history_tokens, 0);
        }
        parent.kvmem_register_append(history_tokens);
        const auto before = parent_state(parent);
        const auto used = parent.kvmem_gpu_page_pool_->used_pages();
        const auto free = parent.kvmem_gpu_page_pool_->free_pages();
        const auto history = parent.block_store_->history_end();
        const auto generation = parent.kvmem_selection_generation();
        require(!parent.make_isolated_generation(UINT32_MAX, UINT32_MAX), "oversized child was admitted");
        require(parent.kvmem_gpu_page_pool_->free_pages() == free, "rejected admission changed pool state");
        Bytes result;
        {
            auto child = parent.make_isolated_generation(input.size() + 16, input.size() + 16);
            require(bool(child), "child admission failed");
            require(child->position() == 0 && !child->block_store() && !child->dflash_executor(),
                    "child inherited parent history or draft");
            auto step = child->forward_n_tokens(input);
            require(step.ok, "child prefill failed");
            for (unsigned i = 0; i < 8; ++i) {
                require(step.argmax_token >= 0, "child produced no argmax");
                const uint32_t token = step.argmax_token;
                const auto *bytes = reinterpret_cast<const uint8_t *>(&token);
                result.insert(result.end(), bytes, bytes + sizeof(token));
                step = child->forward_one_token(token);
                require(step.ok, "child decode failed");
            }
            append(result, device, child->logits_.get());
            require(before == parent_state(parent), "child mutated parent's compute state or KV");
            // An abandoned second child must return its pages, including pages
            // reserved at admission but not yet touched by prefill.
            auto abandoned = parent.make_isolated_generation(64, 128);
            require(bool(abandoned), "second child admission failed");
            require(abandoned->forward_one_token(42).ok, "abandoned child forward failed");
        }
        require(parent.kvmem_gpu_page_pool_->used_pages() == used &&
                parent.kvmem_gpu_page_pool_->free_pages() == free, "child leaked or double-freed pages");
        require(parent.position() == history_tokens && parent.block_store_->history_end() == history &&
                parent.kvmem_selection_generation() == generation && before == parent_state(parent),
                "child teardown changed parent state/history");
        std::cout << "parent_token=" << parent_token << " isolation=exact pages=reclaimed admission=bounded"
                  << (draft ? " dflash=exact" : "") << '\n';
        return result;
    }
};
}
int main(int argc, char **argv) try {
    if (argc < 2 || argc > 3) throw std::invalid_argument("usage: qw3-thinking-summary-isolation-probe MODEL [DFLASH]");
    setenv("QW3_PAGED_KV_PAGE_SIZE", "16", 1); setenv("QW3_KV_DTYPE", "fp8", 1);
    auto device = qw3::make_cuda_device_backend(qw3::LinearBackend::Auto);
    if (!device || !device->begin().ok) throw std::runtime_error("CUDA unavailable");
    {
        qw3::QwenNativeModel model(qw3::open_model_source(argv[1]));
        qw3::QwenWeights weights(model, *device, false, false);
        std::unique_ptr<qw3::DFlashModel> draft_model;
        std::unique_ptr<qw3::DFlashWeights> draft;
        if (argc == 3) {
            draft_model = std::make_unique<qw3::DFlashModel>(argv[2]);
            draft = std::make_unique<qw3::DFlashWeights>(*draft_model, *device);
        }
        qw3::GgufFile file(argv[1]); qw3::QwenTokenizer tokenizer(file);
        const auto ids = [&](const std::string &text) {
            auto encoded = tokenizer.encode(text);
            return std::vector<uint32_t>(encoded.begin(), encoded.end());
        };
        const auto input = qw3::detail::thinking_summary_input(ids(qw3::detail::thinking_summary_prefix()),
            ids("Need to inspect src/cache.py next. Public API must remain unchanged. No tests have run.</think>"),
            tokenizer.token_id("</think>"), ids(qw3::detail::thinking_summary_suffix()));
        auto a = qw3::KvPageTableTestAccess::run(model, weights, *device, input, 100, draft.get());
        auto b = qw3::KvPageTableTestAccess::run(model, weights, *device, input, 900, draft.get());
        qw3::KvPageTableTestAccess::require(a == b, "summary depends on parent history");
        std::cout << "different_parent_histories identical_summary_tokens_and_logits=pass\n";
    }
    if (!device->end().ok) throw std::runtime_error("CUDA end failed");
    return 0;
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
