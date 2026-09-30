#include "dflash_weights.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void check(qw3::DeviceStatus s) { if (!s.ok) throw std::runtime_error(s.message); }
}

// Actual draft projection shapes/weights, with identical activation rows in
// private and joint GEMMs. This isolates batch-dependent arithmetic from the
// scheduler, target features, attention caches and random sampling.
int main(int argc, char **argv) try {
    if (argc != 2) throw std::runtime_error("usage: qw3-dflash-linear-batch DRAFT_DIRECTORY");
    qw3::DFlashModel model(argv[1]);
    auto backend = qw3::make_cuda_device_backend(qw3::LinearBackend::Cublas);
    check(backend->begin());
    qw3::DFlashWeights weights(model, *backend);
    unsigned cases = 0, mismatched_cases = 0;
    const bool exhaustive = std::getenv("QW3_DFLASH_LINEAR_PROBE_ALL_WIDTHS");
    std::vector<std::string> names = {"layers.0.attention_conv.kernel_projection.weight",
                            "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.k_proj.weight",
                            "layers.0.self_attn.o_proj.weight", "layers.0.mlp.gate_proj.weight",
                            "layers.0.mlp.down_proj.weight", "candidate_selector.hidden_projection.weight"};
    if (exhaustive) {
        names = {"candidate_selector.hidden_projection.weight"};
        for (unsigned layer = 0; layer < model.config().num_layers; ++layer)
            for (const char *suffix : {"attention_conv.kernel_projection.weight", "mlp_conv.kernel_projection.weight",
                    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"})
                names.push_back("layers." + std::to_string(layer) + "." + suffix);
    }
    const unsigned maximum_blocks = exhaustive ? 32 : 16;
    const std::vector<unsigned> counts = exhaustive ? std::vector<unsigned>{1,2,3,4,8,16,32} : std::vector<unsigned>{1,2,3,4,16};
    for (const auto &name : names) {
        const auto &weight = weights.at(name);
        const std::vector<uint32_t> widths = exhaustive ? std::vector<uint32_t>{1,2,3,4,5,6,7,8} : std::vector<uint32_t>{1,3,7,8};
        for (uint32_t width : widths) {
            const uint64_t ni = uint64_t(width) * weight.cols, no = uint64_t(width) * weight.rows;
            std::mt19937 rng(1337);
            std::normal_distribution<float> normal(0, .25f);
            std::vector<float> input(ni * maximum_blocks);
            for (auto &v : input) v = normal(rng);
            auto x = backend->tensor_f32(input.size(), "batch-linear-input");
            auto reference = backend->scratch_f32(no * maximum_blocks, "batch-linear-private");
            auto joint = backend->scratch_f32(no * maximum_blocks, "batch-linear-joint");
            check(backend->copy_bytes_from_host(*x, 0, input.data(), input.size() * 4));
            for (unsigned i = 0; i < maximum_blocks; ++i) {
                auto xi = backend->tensor_view(*x, i * ni * 4, ni, 4, qw3::DeviceTensorDType::F32, "private-x");
                auto yi = backend->tensor_view(*reference, i * no * 4, no, 4, qw3::DeviceTensorDType::F32, "private-y");
                check(backend->linear_matmul(*yi, weight, *xi, width, weight.cols, weight.rows));
            }
            std::vector<float> expected(no * maximum_blocks), actual(no * maximum_blocks);
            check(backend->copy_to_host(*reference, expected.data(), 0, expected.size()));
            for (unsigned count : counts) {
                if (std::getenv("QW3_DFLASH_LINEAR_PROBE_GENERIC"))
                    check(backend->linear_matmul(*joint, weight, *x, width * count, weight.cols, weight.rows));
                else
                    check(backend->dflash_linear_blocks(*joint, weight, *x, count, width));
                check(backend->copy_to_host(*joint, actual.data(), 0, no * count));
                uint64_t differences = 0; float maximum = 0;
                for (uint64_t i = 0; i < no * count; ++i) {
                    if (!std::isfinite(actual[i])) throw std::runtime_error("non-finite projection");
                    differences += std::memcmp(&actual[i], &expected[i], 4) != 0;
                    maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
                }
                ++cases; mismatched_cases += differences != 0;
                std::cout << "draft_linear name=" << name << " width=" << width << " requests=" << count
                          << " mismatches=" << differences << " max_abs=" << maximum << '\n';
                if (std::getenv("QW3_DFLASH_LINEAR_PROBE_TIMING") && width >= 7 && (count == 4 || count == 16)) {
                    const char *value = std::getenv("QW3_DFLASH_LT_DRAFT");
                    const std::string original = value ? value : "0";
                    for (unsigned trial = 0; trial < 3; ++trial) {
                        for (unsigned step = 0; step < 2; ++step) {
                            const bool lt = (trial + step) % 2;
                            setenv("QW3_DFLASH_LT_DRAFT", lt ? "1" : "0", 1);
                            check(backend->dflash_linear_blocks(*joint, weight, *x, count, width));
                            check(backend->synchronize());
                            const auto start = std::chrono::steady_clock::now();
                            for (unsigned repeat = 0; repeat < 20; ++repeat)
                                check(backend->dflash_linear_blocks(*joint, weight, *x, count, width));
                            check(backend->synchronize());
                            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 20;
                            std::cout << "draft_linear_timing name=" << name << " width=" << width << " requests=" << count
                                      << " trial=" << trial << " lt=" << lt << " milliseconds=" << ms << '\n';
                        }
                    }
                    setenv("QW3_DFLASH_LT_DRAFT", original.c_str(), 1);
                }
            }
        }
    }
    std::cout << "draft_linear cases=" << cases << " mismatched_cases=" << mismatched_cases << '\n';
    return mismatched_cases ? 1 : 0;
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 2; }
