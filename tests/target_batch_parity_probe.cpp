#include "qwen_executor.hpp"
#include "qw3/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

// Model-dependent diagnostic, not CTest: identical prefix snapshot and teacher
// tokens, no draft calls. All layer residuals locate serial/batch divergence.
namespace {
using namespace qw3;
void check(DeviceStatus s) { if (!s.ok) throw std::runtime_error(s.message); }
uint32_t prediction(const NativeExecutorReport &r) {
    if (!r.ok || r.argmax_token < 0) throw std::runtime_error("target forward failed");
    return r.argmax_token;
}
std::vector<float> features(QwenExecutor &target, DeviceBackend &backend,
                            uint32_t width, uint32_t rows) {
    auto view = target.dflash_features();
    if (!view.data || view.rows != rows) throw std::runtime_error("feature rows missing");
    std::vector<float> result(uint64_t(rows) * width);
    check(backend.copy_to_host(*view.data, result.data(), 0, result.size()));
    return result;
}
}

int main(int argc, char **argv) try {
    if (argc < 2)
        throw std::invalid_argument("usage: qw3-target-batch-parity-probe TARGET [PREFIX_SPLIT=19] [--require-exact] [--history N] [--kvmem]");
    const uint32_t split = argc >= 3 ? std::stoul(argv[2]) : 19;
    bool require_exact = false, kvmem = false, benchmark_widths = false;
    std::vector<uint32_t> benchmark_feature_layers;
    uint32_t history = 0;
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--require-exact") require_exact = true;
        else if (arg == "--kvmem") kvmem = true;
        else if (arg == "--history" && i + 1 < argc) history = std::stoul(argv[++i]);
        else if (arg == "--benchmark-widths") benchmark_widths = true;
        else if (arg == "--feature-layers" && i + 1 < argc) {
            std::istringstream values(argv[++i]);
            std::string value;
            while (std::getline(values, value, ',')) benchmark_feature_layers.push_back(std::stoul(value));
        }
        else throw std::invalid_argument("unknown diagnostic option: " + arg);
    }
    if (history > 131072) throw std::invalid_argument("history exceeds 128K");
    if (benchmark_widths && (kvmem || benchmark_feature_layers.empty()))
        throw std::invalid_argument("width timings require dense teacher prefix and explicit draft feature layers");
    auto backend = make_cuda_device_backend(LinearBackend::Auto);
    check(backend->begin());
    QwenNativeModel model(open_model_source(argv[1]));
    for (uint32_t layer : benchmark_feature_layers)
        if (layer >= model.config().n_layers) throw std::invalid_argument("benchmark feature layer out of range");
    QwenWeights weights(model, *backend, false, false);
    QwenExecutor target(model, weights, *backend, kvmem ? 8192 : std::max(4096u, history + 4096));
    if (kvmem) {
        KvMemStoreConfig cfg;
        cfg.block_tokens = 32; cfg.repository_blocks = 256;
        cfg.select_budget = cfg.prefill_budget = 1024; cfg.gen_budget = 512;
        cfg.sink_blocks = 1; cfg.recent_blocks = 4;
        cfg.immutable_source_k = true; cfg.index_placement = KvMemIndexPlacement::CPU;
        cfg.cpu_tier_bytes = 4ull << 30;
        cfg.gpu_memory_ratio = 0; cfg.estimated_gpu_block_capacity = 128;
        target.configure_kvmem(cfg); target.set_kvmem_enabled(true);
        target.kvmem_gc_begin_step();
    }
    auto tokenizer = std::filesystem::is_directory(argv[1])
        ? std::make_unique<QwenTokenizer>(std::string(argv[1]))
        : std::make_unique<QwenTokenizer>(model.gguf());
    const auto encoded = tokenizer->encode("<|im_start|>user\nCompute 17 * 23 and explain the calculation."
        "<|im_end|>\n<|im_start|>assistant\n<think>\n");
    std::vector<uint32_t> prompt(encoded.begin(), encoded.end());
    if (history) {
        const auto line = tokenizer->encode("Function fetch_record retries network requests. The retry limit is three and the timeout is 500 milliseconds. Error: connection reset by peer.\n");
        std::vector<uint32_t> prefix;
        while (prefix.size() < history) prefix.insert(prefix.end(), line.begin(), line.end());
        prefix.resize(history);
        for (size_t begin = 0; begin < prefix.size(); begin += 256) {
            const auto end = std::min(begin + 256, prefix.size());
            if (!target.forward_n_tokens(std::vector<uint32_t>(prefix.begin() + begin, prefix.begin() + end), false).ok)
                throw std::runtime_error("history prefill failed");
            if (kvmem) target.kvmem_register_append(end - begin);
        }
        if (kvmem) target.kvmem_reselect_prefill_pressure();
    }
    if (split >= prompt.size()) throw std::invalid_argument("split exceeds prefix");
    if (split) {
        auto r = target.forward_n_tokens(std::vector<uint32_t>(prompt.begin(), prompt.begin() + split), false);
        if (!r.ok) throw std::runtime_error("prefix prefill failed");
        if (kvmem) target.kvmem_register_append(split);
    }
    uint32_t next = prediction(target.forward_n_tokens(
        std::vector<uint32_t>(prompt.begin() + split, prompt.end()), true));
    if (kvmem) target.kvmem_register_append(prompt.size() - split);
    std::vector<uint32_t> layers(model.config().n_layers);
    std::iota(layers.begin(), layers.end(), 0);
    target.set_dflash_layers(layers);
    const uint32_t hidden = model.config().n_embd;
    const uint32_t feature_width = hidden * layers.size();
    bool all_exact = true;
    bool all_logits_exact = true;
    for (uint32_t step = 0; step < 4; ++step) {
        const auto base = target.position();
        auto prefix = target.snapshot_state();
        std::vector<uint32_t> inputs, expected;
        std::vector<std::vector<float>> serial_features, serial_logits;
        for (uint32_t row = 0; row < 8; ++row) {
            inputs.push_back(next);
            next = prediction(target.forward_one_token(next));
            expected.push_back(next);
            serial_features.push_back(features(target, *backend, feature_width, 1));
            serial_logits.emplace_back();
            if (!target.copy_last_logits(serial_logits.back()))
                throw std::runtime_error("serial logits missing");
        }
        auto serial_end = target.snapshot_state();
        for (uint32_t rows = 1; rows <= 8; ++rows) {
            target.restore_state(prefix);
            std::vector<DeviceArgmax> predictions;
            std::vector<std::vector<float>> batch_logits;
            QwenExecutor::StateCheckpointSet checkpoints;
            prediction(target.forward_n_tokens(std::vector<uint32_t>(inputs.begin(), inputs.begin() + rows),
                true, &predictions, &checkpoints, rows - 1, false, &batch_logits));
            const auto batch_features = features(target, *backend, feature_width, rows);
            for (uint32_t layer = 0; layer < layers.size(); ++layer) {
                float first_max = 0, all_max = 0;
                for (uint32_t row = 0; row < rows; ++row) for (uint32_t col = 0; col < hidden; ++col) {
                    const auto offset = layer * hidden + col;
                    const float error = std::abs(batch_features[uint64_t(row) * feature_width + offset] -
                                                  serial_features[row][offset]);
                    if (!std::isfinite(error)) throw std::runtime_error("nonfinite residual");
                    all_max = std::max(all_max, error);
                    if (row == 0) first_max = std::max(first_max, error);
                }
                if (all_max) {
                    all_exact = false;
                    std::cout << "layer position=" << base << " rows=" << rows << " layer=" << layer
                              << " first_max_abs=" << first_max << " all_max_abs=" << all_max << '\n';
                }
            }
            float worst = 0;
            uint32_t mismatches = 0;
            for (uint32_t row = 0; row < rows; ++row) {
                mismatches += predictions[row].token != int32_t(expected[row]);
                if (batch_logits[row].size() != serial_logits[row].size())
                    throw std::runtime_error("logit shape mismatch");
                for (size_t v = 0; v < batch_logits[row].size(); ++v) {
                    const float error = std::abs(batch_logits[row][v] - serial_logits[row][v]);
                    if (!std::isfinite(error)) throw std::runtime_error("nonfinite logits");
                    worst = std::max(worst, error);
                }
            }
            all_logits_exact = all_logits_exact && worst == 0 && mismatches == 0;
            std::cout << "logits position=" << base << " rows=" << rows << " mismatches=" << mismatches
                      << " max_abs=" << worst << "\n" << std::flush;
        }
        if (benchmark_widths && step == 0) {
            // Cost attribution only: identical teacher tokens and prefix for
            // every width. Reuse the checkpoint allocation after warmup; keep
            // prefix restore outside the timed interval. Capture the actual
            // drafter's target layers instead of the all-layer parity trace.
            if (!all_exact || !all_logits_exact)
                throw std::runtime_error("refuse width timing after target parity failure");
            target.set_dflash_layers(benchmark_feature_layers);
            constexpr uint32_t iterations = 10;
            const std::vector<uint32_t> widths{8, 4, 6, 2};
            std::vector<QwenExecutor::StateCheckpointSet> timing_checkpoints(widths.size());
            for (uint32_t trial = 0; trial < 4; ++trial) {
                for (uint32_t order = 0; order < widths.size(); ++order) {
                    const uint32_t slot = (trial + order) % widths.size();
                    const uint32_t rows = widths[slot];
                    std::vector<uint32_t> teacher(inputs.begin(), inputs.begin() + rows);
                    std::vector<DeviceArgmax> predictions;
                    double elapsed_ms = 0;
                    // First iteration warms width-specific workspaces.
                    for (uint32_t iteration = 0; iteration <= iterations; ++iteration) {
                        target.restore_state(prefix);
                        check(backend->synchronize());
                        predictions.clear();
                        const auto started = std::chrono::steady_clock::now();
                        prediction(target.forward_n_tokens(teacher, true, &predictions,
                            &timing_checkpoints[slot], rows - 1, /*copy_last_logits=*/false));
                        check(backend->synchronize());
                        const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - started).count();
                        if (predictions.size() != rows) throw std::runtime_error("timed verifier row count changed");
                        for (uint32_t row = 0; row < rows; ++row)
                            if (predictions[row].token != int32_t(expected[row]))
                                throw std::runtime_error("timed teacher argmax changed");
                        if (iteration) elapsed_ms += ms;
                    }
                    std::cout << "target_verify_width_timing history=" << history
                              << " position=" << base << " trial=" << trial
                              << " rows=" << rows << " iterations=" << iterations
                              << " total_ms=" << elapsed_ms
                              << " mean_ms=" << elapsed_ms / iterations
                              << " restore_excluded=true teacher_argmax_exact=true"
                              << " feature_layers=" << benchmark_feature_layers.size()
                              << " dense_prefix_not_kvmem_throughput=true\n" << std::flush;
                }
            }
            target.set_dflash_layers(layers);
        }
        if (kvmem) {
            // Restoring a transient snapshot restores model state, not a
            // discarded repository append transaction. Publish a real replay
            // from the committed boundary before opening the next test block.
            target.restore_state(prefix);
            for (uint32_t token : inputs) {
                prediction(target.forward_one_token(token));
                target.kvmem_register_append(1);
            }
        } else {
            target.restore_state(serial_end);
        }
    }
    check(backend->end());
    std::cout << "diagnostic_complete residuals_bit_exact=" << all_exact
              << " logits_bit_exact=" << all_logits_exact << " (no performance claim)\n";
    if (require_exact && (!all_exact || !all_logits_exact))
        throw std::runtime_error("serial/verifier exactness regression");
    return 0;
} catch (const std::exception &e) {
    std::cerr << "target batch diagnostic failed: " << e.what() << '\n';
    return 1;
}
