#include "qwen_executor.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <unistd.h>

using namespace qw3;

static void check(bool ok, const char *why) {
    if (!ok) throw std::runtime_error(why);
}
static bool near(double a, double b) { return std::abs(a - b) < 1e-7; }
static double score(const detail::KvMemQueryHistory &history, uint64_t start) {
    const auto found = history.scores().find(start);
    return found == history.scores().end() ? 0 : found->second;
}
template<class Exception, class Fn>
static void rejects(Fn fn, const char *why) {
    bool caught = false;
    try { fn(); } catch (const Exception &) { caught = true; }
    check(caught, why);
}

// Exercise production executor bookkeeping/selection with host allocations.
// The controlled score output tests scorer dispatch, not attention numerics;
// all model forward kernels deliberately fail if accidentally invoked.
struct HistoryBackend final : DeviceBackend {
    struct Tensor : DeviceTensor { std::vector<float> values; };
    std::vector<float> next_scores;
    unsigned scorer_calls = 0;
    const char *name() const override { return "query-history-host"; }
    DeviceStatus begin() override { return {}; }
    DeviceStatus end() override { return {}; }
    DeviceStatus synchronize() override { return {}; }
    std::unique_ptr<DeviceTensor> tensor_f32(uint64_t count, const char *) override {
        auto tensor = std::make_unique<Tensor>();
        tensor->count = count;
        tensor->values.resize(count);
        return tensor;
    }
    DeviceStatus copy_to_host(const DeviceTensor &src, float *dst,
                             uint64_t offset, uint64_t count) override {
        const auto &values = dynamic_cast<const Tensor &>(src).values;
        check(offset <= values.size() && count <= values.size() - offset,
              "score read exceeds host tensor");
        std::copy_n(values.data() + offset, count, dst);
        return {};
    }
    DeviceStatus block_attn_score_softmax_pages_device(DeviceTensor &output,
            const DeviceTensor &, const DeviceTensor &, uint32_t layers,
            uint32_t queries, uint32_t, uint32_t blocks, uint32_t,
            uint32_t, uint32_t, uint32_t, float, uint32_t, uint32_t,
            uint32_t, uint32_t, uint32_t accumulate, uint64_t, uint64_t) override {
        check(layers == 1 && queries == 2 && blocks == next_scores.size() &&
              accumulate == 0, "all-fit finalization dispatched an unexpected scorer");
        dynamic_cast<Tensor &>(output).values = next_scores;
        ++scorer_calls;
        return {};
    }
    std::unique_ptr<DeviceWeight> weight_f32(const float *, uint64_t n, const char *) override {
        auto w = std::make_unique<DeviceWeight>(); w->rows = n; w->cols = 1; return w;
    }
    std::unique_ptr<DeviceWeight> weight_q8_0(const void *, uint64_t, uint64_t, const char *) override { throw std::logic_error("unexpected model compute"); }
    DeviceStatus q8_0_get_row(DeviceTensor &, const DeviceWeight &, uint64_t) override { throw std::logic_error("unexpected model compute"); }
    DeviceStatus q8_0_matvec(DeviceTensor &, const DeviceWeight &, const DeviceTensor &) override { throw std::logic_error("unexpected model compute"); }
    DeviceStatus rms_norm(DeviceTensor &, const DeviceTensor &, const DeviceWeight &, float) override { throw std::logic_error("unexpected model compute"); }
    DeviceStatus add(DeviceTensor &, const DeviceTensor &, const DeviceTensor &) override { throw std::logic_error("unexpected model compute"); }
    DeviceStatus silu(DeviceTensor &, const DeviceTensor &) override { throw std::logic_error("unexpected model compute"); }
    DeviceStatus mul(DeviceTensor &, const DeviceTensor &, const DeviceTensor &) override { throw std::logic_error("unexpected model compute"); }
    DeviceStatus recurrent_single_token(DeviceTensor &, DeviceTensor &, DeviceTensor &, DeviceTensor &, const DeviceTensor &, const DeviceTensor &, const DeviceTensor &, const DeviceTensor &, const DeviceWeight &, const DeviceWeight &, const DeviceWeight &, const DeviceWeight &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float) override { throw std::logic_error("unexpected model compute"); }
    DeviceStatus attention_single_token(DeviceTensor &, DeviceTensor &, DeviceTensor &, const DeviceTensor &, const DeviceWeight &, const DeviceWeight &, uint32_t, uint32_t, uint32_t, float) override { throw std::logic_error("unexpected model compute"); }
    DeviceStatus rmsnorm_per_head(DeviceTensor &, const DeviceWeight &, uint32_t, uint32_t, uint32_t, float) override { throw std::logic_error("unexpected model compute"); }
    DeviceStatus rope_partial(DeviceTensor &, uint32_t, uint32_t, uint32_t, uint32_t, float) override { throw std::logic_error("unexpected model compute"); }
    DeviceStatus kv_append(DeviceTensor &, const DeviceTensor &, uint32_t, uint32_t) override { throw std::logic_error("unexpected model compute"); }
    DeviceStatus attention_decode(DeviceTensor &, DeviceTensor &, const DeviceTensor &, uint32_t, const DeviceTensor &, const DeviceTensor &, uint32_t, uint32_t, uint32_t, uint32_t, float) override { throw std::logic_error("unexpected model compute"); }
    DeviceStatus apply_attn_gate(DeviceTensor &, const DeviceTensor &, uint32_t, uint32_t, uint32_t) override { throw std::logic_error("unexpected model compute"); }
    DeviceStatus zero_tensor(DeviceTensor &) override { throw std::logic_error("unexpected model compute"); }
    DeviceArgmax argmax(const DeviceTensor &) override { throw std::logic_error("unexpected model compute"); }
};

struct HistoryModelSource final : ModelSource {
    std::string directory;
    ModelSourceInfo description{"history-test", "qwen35", 1, 4};
    mutable std::map<std::string, ModelTensorInfo> tensors;
    float value = 0;
    explicit HistoryModelSource(std::string dir) : directory(std::move(dir)) {}
    const ModelTensorInfo *find_tensor(const std::string &name) const override {
        auto &t = tensors[name]; t.name = name; t.source_name = name;
        t.type = ModelTensorType::F32; t.dims = {1}; t.data = &value; t.bytes = sizeof(value);
        return &t;
    }
    const ModelSourceInfo &info() const override { return description; }
    const std::string &model_directory() const override { return directory; }
};

namespace qw3 {
// Reuse the established storage-test access point. No new production test-only
// accessors or alternate implementations of the history lifecycle are needed.
struct KvPageTableTestAccess {
    static KvMemStoreConfig config(uint32_t budget = 32) {
        KvMemStoreConfig cfg;
        cfg.block_tokens = 32;
        cfg.select_budget = budget;
        cfg.sink_blocks = cfg.recent_blocks = 0;
        cfg.query_history = true;
        return cfg;
    }
    static void ready_query(QwenExecutor &e) {
        e.kvmem_enabled_ = true;
        e.kvmem_query_begin_ = 96;
        e.kvmem_query_end_ = 98;
        e.kvmem_query_span_ = 2;
        e.g_query_multi_count_ = 2;
        e.g_query_multi_ready_ = true;
    }
    static void publish(QwenExecutor &e, const std::vector<double> &scores,
                        uint32_t blocks = 4) {
        e.kvmem_store_scores(scores);
        e.kvmem_query_history_observe(scores, blocks);
    }

    static void selection_and_replay(const QwenNativeModel &model,
                                    const QwenWeights &weights, HistoryBackend &backend) {
        QwenExecutor e(model, weights, backend, 256);
        e.block_store_ = std::make_unique<KvMemStore>(config());
        e.block_store_->register_append(128);
        ready_query(e);
        detail::KvMemQueryHistory history;
        history.begin(128, 0, .5);
        history.preview({{0, 1}});
        history.commit();
        e.kvmem_query_history_begin(history, 256, 128, .5);
        rejects<std::logic_error>([&] { e.kvmem_query_history_begin(history, 256, 128, .5); },
                                  "nested logical requests were accepted");
        publish(e, {.4, .6, 0, 0});
        check(e.block_store_->pick_topk_blocks() == std::vector<uint32_t>{1},
              "fixture does not separate instantaneous and historical ranking");
        check(e.kvmem_selection_with_pin() == std::vector<uint32_t>{0},
              "query history did not enter the production selection path");
        const auto candidate = history.scores();
        for (unsigned replay = 0; replay < 8; ++replay) {
            publish(e, {.4, .6, 0, 0});
            check(history.scores() == candidate && history.round_count() == 1,
                  "replayed scorer added another decay/reward or committed early");
        }
        rejects<std::logic_error>([&] { e.kvmem_query_history_observe({1}, 1); },
                                  "score vector from a different block domain was accepted");
        e.kvmem_query_history_finalize_input();
        check(history.round_count() == 2 && history.scores() == candidate,
              "input finalization failed to commit exactly one candidate");
        e.kvmem_query_history_finalize_input();
        publish(e, {0, 0, 1, 0});
        check(history.round_count() == 2 && history.scores() == candidate &&
              e.kvmem_selection_with_pin() == std::vector<uint32_t>{2},
              "decode refresh changed input history or lost instantaneous ranking");

        e.kvmem_query_history_begin(history, 256, 256, .5);
        publish(e, {.4, .6, 0, 0});
        check(history.scores() == candidate, "executor regeneration rewarded itself twice");
        e.kvmem_query_history_abort();
        check(!history.open() && history.scores() == candidate,
              "executor abort discarded previously committed scores");

        // A complete zero frame supplies no new evidence. The prior winner is
        // deliberately old so a raw-zero recency fallback cannot pass here.
        e.kvmem_query_history_begin(history, 320, 256, .5);
        publish(e, {0, 0, 0, 0});
        check(e.block_store_->pick_topk_blocks() != std::vector<uint32_t>{0},
              "zero-frame fixture failed to distinguish recency from retained history");
        check(!history.has_observation() && history.scores() == candidate &&
              e.kvmem_selection_with_pin() == std::vector<uint32_t>{0},
              "complete zero frame discarded or decayed its parent's ranking");
        e.kvmem_query_history_finalize_input();
        e.kvmem_query_history_finalize_input();
        check(history.round_count() == 3 && !history.has_observation() &&
              history.scores() == candidate,
              "zero-evidence input aged history or committed more than once");

        detail::KvMemQueryHistory cold;
        e.kvmem_query_history_begin(cold, 128, 0, .5);
        publish(e, {0, 0, 0, 0});
        check(!e.kvmem_query_history_selection() && !cold.has_observation() &&
              e.kvmem_selection_with_pin() == e.block_store_->pick_topk_blocks(),
              "empty cold history changed the original raw-zero selection fallback");
        e.kvmem_query_history_finalize_input();
        check(cold.round_count() == 1 && cold.scores().empty(),
              "cold zero frame manufactured historical evidence");
    }

    static void guards_and_boundaries(const QwenNativeModel &model,
                                     const QwenWeights &weights, HistoryBackend &backend) {
        QwenExecutor e(model, weights, backend, 256);
        e.block_store_ = std::make_unique<KvMemStore>(config());
        e.block_store_->register_append(128);
        ready_query(e);
        detail::KvMemQueryHistory history;
        e.kvmem_query_history_begin(history, 90, 0, .5);
        e.g_query_multi_count_ = 1;
        publish(e, {1, 1, 1, 1});
        check(!history.has_observation() && !e.kvmem_query_history_selection(),
              "incomplete Query capture published history");
        rejects<std::runtime_error>([&] { e.kvmem_query_history_finalize_input(); },
                                    "incomplete input capture was committed");
        e.g_query_multi_count_ = 2;
        for (bool *guard : {&e.kvmem_guided_query_probe_active_,
                            &e.kvmem_query_attention_probe_active_,
                            &e.kvmem_provisional_prefill_, &e.kvmem_blend_active_}) {
            *guard = true;
            publish(e, {1, 1, 1, 1});
            check(!history.has_observation() && !e.kvmem_query_history_selection(),
                  "private/provisional/blend scoring entered logical history");
            rejects<std::logic_error>([&] { e.kvmem_query_history_finalize_input(); },
                                      "provisional input was committed");
            *guard = false;
        }
        publish(e, {3, 1, 100, 100});
        check(near(score(history, 0), .75) && near(score(history, 32), .25) &&
              score(history, 64) == 0 && score(history, 96) == 0,
              "block overlapping the accepted-input boundary entered history");
        for (bool *guard : {&e.kvmem_query_replay_active_, &e.kvmem_pending_reselect_}) {
            *guard = true;
            rejects<std::logic_error>([&] { e.kvmem_query_history_finalize_input(); },
                                      "unfinished replay/selection was committed");
            *guard = false;
        }
        const auto complete = history.scores();
        e.kvmem_guided_query_probe_active_ = true;
        publish(e, {0, 0, 0, 1});
        e.kvmem_guided_query_probe_active_ = false;
        check(history.scores() == complete && e.kvmem_query_history_selection(),
              "private scorer overwrote a completed public Query observation");
        e.kvmem_query_history_finalize_input();

        // A physical replay/reset of KV must not erase backend-owned history.
        e.reset_state();
        check(history.scores() == complete && history.round_count() == 1,
              "physical executor reset erased logical evidence");
        ready_query(e);
        e.kvmem_query_history_begin(history, 160, 90, .5);
        e.kvmem_query_begin_ = e.kvmem_query_end_ = 0;
        e.kvmem_query_history_finalize_input();
        check(history.scores() == complete && history.round_count() == 2,
              "an input without Query evidence decayed history");
    }

    static void all_fit_finalize(const QwenNativeModel &model,
                                 const QwenWeights &weights, HistoryBackend &backend) {
        QwenExecutor e(model, weights, backend, 256);
        e.block_store_ = std::make_unique<KvMemStore>(config(128));
        e.block_store_->register_append(128);
        ready_query(e);
        e.kvmem_qc_num_layers_ = 1;
        e.g_kbar_multi_ready_ = true;
        e.g_kbar_multi_blocks_ = e.g_kbar_global_capacity_ = 4;
        e.g_kbar_multi_ = backend.tensor_f32(16, "all_fit_kbar");
        e.g_query_multi_ = backend.tensor_f32(8, "all_fit_query");
        e.g_score_dev_ = backend.tensor_f32(4, "all_fit_scores");
        backend.next_scores = {1, .5f, 0, .5f};
        const auto prior_calls = backend.scorer_calls;
        detail::KvMemQueryHistory history;
        e.kvmem_query_history_begin(history, 128, 0, .5);
        e.kvmem_query_history_finalize_input();
        check(backend.scorer_calls == prior_calls + 1 && history.round_count() == 1 &&
              near(score(history, 0), .5) && near(score(history, 32), .25) &&
              near(score(history, 96), .25),
              "all-fit input failed to obtain a final Query observation");
        check(std::none_of(e.block_store_->blocks().begin(), e.block_store_->blocks().end(),
                           [](const auto &block) { return block.in_working_set; }),
              "all-fit history finalization unexpectedly installed a KV window");
        e.kvmem_query_history_finalize_input();
        check(backend.scorer_calls == prior_calls + 1,
              "repeated finalize rescored an already committed input");

        // Finalization must not reuse an earlier preview after a later scorer
        // publishes another (possibly provisional/fallback) raw score vector.
        e.kvmem_query_history_begin(history, 160, 128, .5);
        publish(e, {1, 0, 0, 0});
        e.kvmem_store_scores({0, 0, 0, 1});
        check(!e.kvmem_query_history_selection(), "stale preview survived raw-score replacement");
        backend.next_scores = {0, 2, 0, 0};
        e.kvmem_query_history_finalize_input();
        check(backend.scorer_calls == prior_calls + 2 &&
              near(score(history, 0), .25) && near(score(history, 32), 1.125),
              "final rescore used stale candidate or double-counted an intermediate preview");
    }

    static void repository_identity(const QwenNativeModel &model,
                                    const QwenWeights &weights, HistoryBackend &backend) {
        QwenExecutor e(model, weights, backend, 256);
        auto cfg = config(); cfg.repository_blocks = 4;
        auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{4, 32});
        e.block_store_ = std::make_unique<KvMemStore>(cfg, repo);
        ready_query(e);
        constexpr uint64_t base = uint64_t{1} << 34;
        std::vector<KvMemBlockHandle> handles;
        std::vector<KvMemIndexStamp> index(4);
        for (uint32_t i = 0; i < 4; ++i) {
            const uint64_t start = base + (i < 2 ? i * 32 : 160 + (i - 2) * 32);
            auto lease = repo->try_reserve(start, 32).commit();
            handles.push_back(lease.handle());
            index[lease.handle().slot] = {lease.handle(), 32};
            e.block_store_->register_repository_block(std::move(lease), i * 32);
        }
        // Historical candidates include an off-GPU block; selection must not
        // reduce the Query domain to the currently resident working set.
        e.block_store_->set_block_tier(handles[0].slot, KvTier::CPU, 0, -1,
                                      handles[0].generation);
        auto frame = [&] {
            e.kvmem_repository_score_frame_ = e.block_store_->repository_score_frame(
                index, base + 128, false);
        };
        detail::KvMemQueryHistory history;
        history.begin(base + 64, 0, .5);
        history.preview({{base, 1}});
        history.commit();
        e.kvmem_query_history_begin(history, base + 128, base + 64, .5);
        frame();
        std::vector<double> raw(4, 1000);
        raw[handles[0].slot] = .4;
        raw[handles[1].slot] = .6;
        publish(e, raw);
        check(near(score(history, base), .9) && near(score(history, base + 32), .6) &&
              score(history, 0) == 0 && history.scores().size() == 2,
              "repository scores used physical slot offsets or included masked future slots");
        check(e.kvmem_selection_with_pin() == std::vector<uint32_t>{handles[0].slot},
              "repository history failed to recall an off-GPU logical block");

        e.kvmem_repository_score_frame_ = {};
        const auto old = handles[0];
        e.block_store_->set_block_tier(old.slot, KvTier::CPU, 0, -1, old.generation);
        check(repo->retire(old) &&
              repo->try_reclaim(old, [](const auto &) { return true; }) ==
                  KvMemRepository::ReclaimResult::Reclaimed,
              "fixture failed to reclaim the scored repository slot");
        auto replacement = repo->try_reserve(base + 96, 32).commit();
        check(replacement.handle().slot == old.slot &&
              replacement.handle().generation != old.generation,
              "fixture did not recycle the original slot");
        handles[0] = replacement.handle();
        index[handles[0].slot] = {handles[0], 32};
        e.block_store_->register_repository_block(std::move(replacement), 0);
        rejects<std::logic_error>([&] { e.kvmem_query_history_selection(); },
                                  "stale observed scores were applied to a reused generation");
        frame();
        publish(e, raw);
        const auto *selected_scores = e.kvmem_query_history_selection();
        check(selected_scores && near((*selected_scores)[handles[0].slot], .4) &&
              near(score(history, base), .5) && near(score(history, base + 96), .4),
              "reused slot inherited old occupant reward or re-observation doubled decay");
        e.kvmem_query_history_finalize_input();
    }
};
} // namespace qw3

int main() {
    const auto directory = std::filesystem::temp_directory_path() /
        ("qw3-query-history-executor-" + std::to_string(getpid()));
    std::filesystem::create_directories(directory);
    std::ofstream(directory / "config.json") << R"({"text_config":{"model_type":"qwen3_5_text",
      "num_hidden_layers":1,"hidden_size":4,"intermediate_size":4,"max_position_embeddings":256,
      "num_attention_heads":1,"num_key_value_heads":1,"head_dim":4,"linear_conv_kernel_dim":4,
      "linear_num_key_heads":1,"linear_num_value_heads":1,"linear_key_head_dim":4,"linear_value_head_dim":4,
      "full_attention_interval":1,"vocab_size":4}})";
    try {
        HistoryBackend backend;
        QwenNativeModel model(std::make_unique<HistoryModelSource>(directory.string()));
        QwenWeights weights(model, backend, false, false);
        KvPageTableTestAccess::selection_and_replay(model, weights, backend);
        KvPageTableTestAccess::guards_and_boundaries(model, weights, backend);
        KvPageTableTestAccess::all_fit_finalize(model, weights, backend);
        KvPageTableTestAccess::repository_identity(model, weights, backend);
    } catch (...) { std::filesystem::remove_all(directory); throw; }
    std::filesystem::remove_all(directory);
    std::cout << "query history executor: selection, replay, provisional guards, all-fit "
                 "finalization and repository generation mapping passed (host bookkeeping only)\n";
}
