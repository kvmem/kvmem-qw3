#include "qw3/device_backend.hpp"
#include "../src/q8_verification_math.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace qw3;
void check(DeviceStatus s) { if (!s.ok) throw std::runtime_error(s.message); }
std::vector<float> random_values(size_t n, unsigned seed, float scale = 1.f) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> v(n);
    for (auto &x : v) x = dist(gen);
    return v;
}
void upload(DeviceBackend &d, DeviceTensor &t, const std::vector<float> &v) {
    check(d.copy_bytes_from_host(t, 0, v.data(), v.size() * sizeof(float)));
}
std::vector<float> read(DeviceBackend &d, const DeviceTensor &t) {
    std::vector<float> v(t.count);
    if (t.dtype == DeviceTensorDType::BF16) {
        std::vector<__nv_bfloat16> raw(t.count);
        check(d.copy_bytes_to_host(t, raw.data(), 0, t.count * sizeof(__nv_bfloat16)));
        for (size_t i = 0; i < v.size(); ++i) v[i] = __bfloat162float(raw[i]);
    } else {
        check(d.copy_to_host(t, v.data(), 0, t.count));
    }
    return v;
}
float compare(const std::vector<float> &a, const std::vector<float> &b,
              const char *name, float tolerance = 0.f) {
    if (a.size() != b.size()) throw std::runtime_error("size mismatch");
    float error = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
            throw std::runtime_error("nonfinite kernel output");
        error = std::max(error, std::abs(a[i] - b[i]));
    }
    if (error > tolerance) {
        std::fprintf(stderr, "%s max_abs=%g tolerance=%g\n", name, error, tolerance);
        throw std::runtime_error(name);
    }
    return error;
}
std::unique_ptr<DeviceWeight> q8_weight(DeviceBackend &d, unsigned rows, unsigned cols, unsigned seed) {
    auto values = random_values(size_t(rows) * cols, seed, .1f);
    std::vector<unsigned char> packed(size_t(rows) * cols / 32 * 34);
    for (size_t block = 0; block < values.size() / 32; ++block) {
        float amax = 0;
        for (unsigned j = 0; j < 32; ++j) amax = std::max(amax, std::abs(values[block * 32 + j]));
        const float scale = amax / 127;
        const __half h = __float2half(scale);
        std::memcpy(packed.data() + block * 34, &h, 2);
        for (unsigned j = 0; j < 32; ++j) {
            const int8_t q = static_cast<int8_t>(std::lrint(values[block * 32 + j] / scale));
            std::memcpy(packed.data() + block * 34 + 2 + j, &q, 1);
        }
    }
    return d.weight_q8_0(packed.data(), rows, cols, "verify test Q8");
}
void linear(DeviceBackend &d, unsigned cols, unsigned batch, bool stable = false,
            bool contiguous = false, unsigned rows = 37) {
    const unsigned stride = contiguous ? rows : rows + 7;
    detail::Q8VerificationMath math(d, stable);
    constexpr float guard = -1024; // Exactly representable in FP32 and BF16.
    auto wg = q8_weight(d, rows, cols, 42), wu = q8_weight(d, rows, cols, 83);
    auto x_host = random_values(size_t(batch) * cols, 71);
    auto x = d.tensor_f32(x_host.size(), "batched x"); upload(d, *x, x_host);
    auto one_x = d.tensor_f32(cols, "serial x");
    auto one = d.tensor_f32(rows, "serial out");
    auto one_bf = d.tensor_bf16(rows, "serial internal residual");
    auto out = d.tensor_f32(size_t(batch) * stride, "guarded out");
    auto out_bf = d.tensor_bf16(size_t(batch) * stride, "guarded internal residual");
    auto scratch = d.tensor_bf16(size_t(batch) * stride, "residual scratch");
    std::vector<float> expected_linear(batch * stride, guard), expected_silu(expected_linear), expected_add(expected_linear);
    auto residual = random_values(batch * stride, 99);
    std::vector<__nv_bfloat16> residual_bf(batch * stride);
    for (unsigned b = 0; b < batch; ++b) for (unsigned j = 0; j < stride; ++j)
        residual_bf[b * stride + j] = __float2bfloat16(j < rows ? residual[b * stride + j] : guard);
    for (unsigned b = 0; b < batch; ++b) {
        check(d.copy_bytes_from_host(*one_x, 0, x_host.data() + b * cols, cols * 4));
        check(d.q8_0_matvec(*one, *wg, *one_x));
        auto values = read(d, *one); std::copy(values.begin(), values.end(), expected_linear.begin() + b * stride);
        check(d.q8_0_matvec_silu_mul(*one, *wg, *wu, *one_x));
        values = read(d, *one); std::copy(values.begin(), values.end(), expected_silu.begin() + b * stride);
        check(d.copy_bytes_from_host(*one_bf, 0, residual_bf.data() + b * stride, rows * 2));
        check(d.q8_0_matvec_add(*one_bf, *wg, *one_x));
        values = read(d, *one_bf); std::copy(values.begin(), values.end(), expected_add.begin() + b * stride);
    }
    upload(d, *out, std::vector<float>(batch * stride, guard));
    DeviceTensor *outs[] = {out.get()}; const DeviceWeight *weights[] = {wg.get()}; const uint32_t strides[] = {stride};
    check(math.fanout(outs, weights, strides, 1, *x, batch, cols));
    compare(read(d, *out), expected_linear, "Q8 fanout differs from serial / overwrote row guard");
    if (stable) {
        upload(d, *out, std::vector<float>(batch * stride, guard));
        check(math.matmul(*out, *wg, *x, batch, cols, stride));
        compare(read(d, *out), expected_linear, "tiled Q8 matmul differs from serial / overwrote row guard");
    }
    upload(d, *out, std::vector<float>(batch * stride, guard));
    check(math.silu_mul(*out, *wg, *wu, *x, batch, cols, stride));
    compare(read(d, *out), expected_silu, "Q8 SwiGLU differs from serial / overwrote row guard");
    check(d.copy_bytes_from_host(*out_bf, 0, residual_bf.data(), residual_bf.size() * 2));
    check(math.add(*out_bf, *out_bf, *scratch, *wg, *x, batch, cols, stride));
    compare(read(d, *out_bf), expected_add, "Q8 residual differs from serial / overwrote row guard");
    std::printf("Q8 linear rows=%u cols=%u batch=%u stable=%d contiguous=%d bit_exact=1 guards=ok\n", rows, cols, batch, stable, contiguous);
}
void wide_fanout(DeviceBackend &d, unsigned cols, unsigned batch, unsigned count, bool padded,
                 bool mma = false) {
    constexpr unsigned row_counts[] = {37, 5, 63, 1};
    constexpr float guard = -1024;
    const unsigned input_stride = cols + (padded ? 11 : 0);
    auto values = random_values(size_t(batch) * input_stride, 152);
    auto x = d.tensor_f32(values.size(), "wide guarded input"); upload(d, *x, values);
    auto one_x = d.tensor_f32(cols, "wide serial input");
    std::vector<std::unique_ptr<DeviceWeight>> weights;
    std::vector<std::unique_ptr<DeviceTensor>> outputs;
    std::vector<const DeviceWeight *> wp;
    std::vector<DeviceTensor *> op;
    std::vector<uint32_t> strides;
    std::vector<std::vector<float>> expected;
    std::vector<std::vector<float>> mma_independent;
    for (unsigned i = 0; i < count; ++i) {
        const unsigned rows = row_counts[i], stride = rows + 7 + i;
        weights.push_back(q8_weight(d, rows, cols, 100 + i)); wp.push_back(weights.back().get());
        outputs.push_back(d.tensor_f32(uint64_t(batch) * stride, "wide guarded output"));
        op.push_back(outputs.back().get()); strides.push_back(stride);
        expected.emplace_back(size_t(batch) * stride, guard);
        if (mma) mma_independent.emplace_back(size_t(batch) * stride, guard);
        upload(d, *op.back(), expected.back());
        auto one = d.tensor_f32(rows, "wide serial output");
        for (unsigned b = 0; b < batch; ++b) {
            check(d.copy_bytes_from_host(*one_x, 0, values.data() + b * input_stride, cols * 4));
            check(d.q8_0_matvec(*one, *wp.back(), *one_x));
            const auto reference = read(d, *one);
            std::copy(reference.begin(), reference.end(), expected.back().begin() + b * stride);
            if (mma) {
                detail::Q8VerificationMath independent(d, true, true, false, true);
                check(independent.matmul(*one, *wp.back(), *one_x, 1, cols, rows));
                if (independent.mma_calls() != 1) throw std::runtime_error("single-row MMA dispatch missing");
                const auto actual = read(d, *one);
                std::copy(actual.begin(), actual.end(), mma_independent.back().begin() + b * stride);
            }
        }
    }
    detail::Q8VerificationMath math(d, true, true, false, mma);
    check(math.fanout(op.data(), wp.data(), strides.data(), count, *x, batch, input_stride));
    if (mma && math.mma_calls() != count) throw std::runtime_error("batched MMA dispatch missing");
    float error = 0;
    for (unsigned i = 0; i < count; ++i) {
        const auto actual = read(d, *op[i]);
        error = std::max(error, compare(actual, expected[i], "wide fanout numerical/guard failure", mma ? 1e-4f : 0.f));
        if (mma) {
            compare(actual, mma_independent[i], "MMA fanout depends on batch membership");
            if (std::memcmp(actual.data(), mma_independent[i].data(), actual.size()*sizeof(float)))
                throw std::runtime_error("MMA fanout batch is not bit exact");
        }
    }
    if (mma)
        std::printf("Q8 MMA fanout cols=%u batch=%u count=%u padded_input=%d max_abs=%g batch_exact=1 guards=ok\n",
                    cols, batch, count, padded, error);
    else
        std::printf("Q8 wide fanout cols=%u batch=%u count=%u padded_input=%d bit_exact=1 guards=ok\n",
                    cols, batch, count, padded);
}
void small_mma_dispatch(DeviceBackend &d) {
    struct Shape { unsigned rows, cols; };
    const char *saved_env = std::getenv("QW3_EXPERIMENTAL_Q8_VERIFY_MMA_SMALL");
    const bool had_env = saved_env != nullptr;
    const std::string saved = saved_env ? saved_env : "";
    for (const Shape shape : {Shape{5120, 6144}, Shape{5120, 17408}, Shape{17408, 5120}}) {
        auto weight = q8_weight(d, shape.rows, shape.cols, 179);
        for (unsigned batch : {1u, 2u, 7u, 8u, 9u, 16u, 65u}) {
            const unsigned in_stride = shape.cols + 11, out_stride = shape.rows + 7;
            auto x = d.tensor_f32(uint64_t(batch) * in_stride, "small MMA strided input");
            upload(d, *x, random_values(x->count, 341 + batch));
            auto out = d.tensor_f32(uint64_t(batch) * out_stride, "small MMA guarded output");
            const std::vector<float> guards(out->count, -1024);
            detail::Q8VerificationMath math(d, true, true, false, true);
            setenv("QW3_EXPERIMENTAL_Q8_VERIFY_MMA_SMALL", "0", 1);
            upload(d, *out, guards);
            check(math.matmul(*out, *weight, *x, batch, in_stride, out_stride));
            const auto reference = read(d, *out);
            setenv("QW3_EXPERIMENTAL_Q8_VERIFY_MMA_SMALL", "1", 1);
            upload(d, *out, guards);
            check(math.matmul(*out, *weight, *x, batch, in_stride, out_stride));
            const auto actual = read(d, *out);
            if (std::memcmp(actual.data(), reference.data(), actual.size() * sizeof(float)))
                throw std::runtime_error("small MMA integrated dispatch changed output or guards");
            for (unsigned b = 0; b < batch; ++b)
                for (unsigned col = shape.rows; col < out_stride; ++col)
                    if (actual[size_t(b) * out_stride + col] != -1024)
                        throw std::runtime_error("small MMA integrated dispatch overwrote padding");
            std::printf("Q8 small MMA dispatch rows=%u cols=%u batch=%u bit_exact=1 guards=ok\n",
                        shape.rows, shape.cols, batch);
        }
    }
    if (had_env) setenv("QW3_EXPERIMENTAL_Q8_VERIFY_MMA_SMALL", saved.c_str(), 1);
    else unsetenv("QW3_EXPERIMENTAL_Q8_VERIFY_MMA_SMALL");
}
void benchmark_wide(DeviceBackend &d) {
    if (!d.supports_q8_verify_fanout()) throw std::runtime_error("benchmark requires QW3_Q8_VERIFY_WIDE=1");
    struct Shape { unsigned rows, cols, count; };
    for (const Shape shape : {Shape{5120, 17408, 1}, Shape{17408, 5120, 2}, Shape{5120, 5120, 4}}) {
        std::vector<std::unique_ptr<DeviceWeight>> weights;
        std::vector<const DeviceWeight *> wp;
        for (unsigned i = 0; i < shape.count; ++i) {
            weights.push_back(q8_weight(d, shape.rows, shape.cols, 210 + i));
            wp.push_back(weights.back().get());
        }
        for (unsigned batch : {16u, 32u, 64u, 128u}) {
            auto x = d.tensor_f32(uint64_t(batch) * shape.cols, "wide timing input");
            upload(d, *x, random_values(x->count, 333));
            std::vector<std::unique_ptr<DeviceTensor>> outputs;
            std::vector<DeviceTensor *> op;
            std::vector<uint32_t> strides(shape.count, shape.rows);
            for (unsigned i = 0; i < shape.count; ++i) {
                outputs.push_back(d.tensor_f32(uint64_t(batch) * shape.rows, "wide timing output"));
                op.push_back(outputs.back().get());
            }
            std::vector<std::vector<float>> reference;
            for (bool wide : {false, true}) {
                detail::Q8VerificationMath math(d, true, wide);
                check(math.fanout(op.data(), wp.data(), strides.data(), shape.count, *x, batch, shape.cols));
                for (unsigned i = 0; i < shape.count; ++i) {
                    auto actual = read(d, *op[i]);
                    if (!wide) reference.push_back(std::move(actual));
                    else compare(actual, reference[i], "timing inputs differ between wide and reference");
                }
            }
            for (unsigned trial = 0; trial < 3; ++trial) {
                double ms[2] = {};
                for (unsigned step = 0; step < 2; ++step) {
                    const bool wide = (step + trial) % 2;
                    detail::Q8VerificationMath math(d, true, wide);
                    for (unsigned warm = 0; warm < 3; ++warm)
                        check(math.fanout(op.data(), wp.data(), strides.data(), shape.count, *x, batch, shape.cols));
                    check(d.synchronize());
                    const auto start = std::chrono::steady_clock::now();
                    for (unsigned repeat = 0; repeat < 20; ++repeat)
                        check(math.fanout(op.data(), wp.data(), strides.data(), shape.count, *x, batch, shape.cols));
                    check(d.synchronize());
                    ms[wide] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 20;
                }
                std::printf("Q8 wide timing rows=%u cols=%u count=%u batch=%u trial=%u reference_ms=%.6f wide_ms=%.6f speedup=%.6f bit_exact=1 scope=warm_operator\n",
                    shape.rows, shape.cols, shape.count, batch, trial, ms[0], ms[1], ms[0] / ms[1]);
                std::fflush(stdout);
            }
        }
    }
}
void wide_swiglu(DeviceBackend &d, unsigned cols, unsigned features, bool padded) {
    if (!d.supports_q8_verify_fanout()) throw std::runtime_error("wide SwiGLU test requires QW3_Q8_VERIFY_WIDE=1");
    auto gate = q8_weight(d, features, cols, 871), up = q8_weight(d, features, cols, 872);
    constexpr float guard = -1024;
    const unsigned input_stride = cols + (padded ? 11 : 0);
    const unsigned output_stride = features + (padded ? 7 : 0);
    for (unsigned rows : {8u, 9u, 16u, 31u, 32u, 33u, 64u}) {
        auto x = d.tensor_f32(uint64_t(rows) * input_stride, "wide SwiGLU input");
        upload(d, *x, random_values(x->count, 873));
        const uint64_t capacity = uint64_t(rows + 1) * output_stride;
        auto out = d.tensor_f32(capacity, "wide SwiGLU guarded output");
        auto g = d.tensor_f32(capacity, "wide SwiGLU guarded gate");
        auto u = d.tensor_f32(capacity, "wide SwiGLU guarded up");
        const std::vector<float> guards(capacity, guard);
        upload(d, *out, guards); upload(d, *g, guards); upload(d, *u, guards);
        detail::Q8VerificationMath reference(d, true, false), candidate(d, true, true);
        check(reference.silu_mul(*out, *gate, *up, *x, rows, input_stride, output_stride));
        const auto expected = read(d, *out);
        upload(d, *out, guards);
        check(candidate.silu_mul(*out, *gate, *up, *x, rows, input_stride, output_stride, g.get(), u.get()));
        if (candidate.wide_swiglu_calls() != unsigned(rows > 8 && !padded))
            throw std::runtime_error("wide SwiGLU dispatch did not match the qualified shape");
        const auto actual = read(d, *out);
        compare(actual, expected, "wide SwiGLU differs from fused reference / overwrote guard");
        if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)))
            throw std::runtime_error("wide SwiGLU is not bit exact");
        const auto gates = read(d, *g), ups = read(d, *u);
        for (uint64_t i = uint64_t(rows) * output_stride; i < capacity; ++i)
            if (gates[i] != guard || ups[i] != guard) throw std::runtime_error("wide SwiGLU overwrote scratch tail");
        std::printf("Q8 wide SwiGLU features=%u cols=%u rows=%u padded_fallback=%d bit_exact=1 guards=ok\n",
                    features, cols, rows, padded);
    }
}
void wide_add(DeviceBackend &d, unsigned cols, unsigned features, bool padded) {
    if (!d.supports_q8_verify_fanout()) throw std::runtime_error("wide add requires QW3_Q8_VERIFY_WIDE=1");
    auto weight = q8_weight(d, features, cols, 901);
    constexpr float guard = -1024;
    const unsigned input_stride = cols + (padded ? 11 : 0);
    const unsigned output_stride = features + (padded ? 7 : 0);
    for (unsigned rows : {8u, 9u, 16u, 24u, 31u, 32u, 33u}) {
        auto x = d.tensor_f32(uint64_t(rows) * input_stride, "wide add input");
        upload(d, *x, random_values(x->count, 902));
        const uint64_t capacity = uint64_t(rows + 1) * output_stride;
        auto out = d.tensor_f32(capacity, "wide add output");
        auto scratch = d.tensor_f32(capacity, "wide add scratch");
        auto residual = d.tensor_f32(capacity, "wide add residual");
        const std::vector<float> guards(capacity, guard);
        auto values = guards;
        auto data = random_values(uint64_t(rows) * features, 903);
        for (unsigned row = 0; row < rows; ++row)
            std::copy_n(data.data() + uint64_t(row) * features, features,
                        values.data() + uint64_t(row) * output_stride);
        for (bool in_place : {false, true}) {
            upload(d, *residual, values); upload(d, *out, values); upload(d, *scratch, guards);
            detail::Q8VerificationMath reference(d, true, false), candidate(d, true, true, true);
            const auto &r = in_place ? *out : *residual;
            check(reference.add(*out, r, *scratch, *weight, *x, rows, input_stride, output_stride));
            const auto expected = read(d, *out);
            upload(d, *out, values); upload(d, *scratch, guards);
            check(candidate.add(*out, r, *scratch, *weight, *x, rows, input_stride, output_stride));
            if (candidate.wide_add_calls() != unsigned(rows > 8 && !padded && in_place))
                throw std::runtime_error("wide add dispatch mismatch");
            const auto actual = read(d, *out), temp = read(d, *scratch);
            compare(actual, expected, "wide add differs from tiled reference / overwrote guard");
            if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)))
                throw std::runtime_error("wide add is not bit exact");
            compare(read(d, *residual), values, "wide add modified separate residual");
            for (uint64_t i = uint64_t(rows) * output_stride; i < capacity; ++i)
                if (temp[i] != guard) throw std::runtime_error("wide add overwrote scratch tail");
            std::printf("Q8 wide add features=%u cols=%u rows=%u padded_fallback=%d inplace=%d bit_exact=1 guards=ok\n",
                        features, cols, rows, padded, in_place);
        }
    }
}
void benchmark_wide_add(DeviceBackend &d) {
    constexpr unsigned features = 5120, cols = 6144;
    auto weight = q8_weight(d, features, cols, 904);
    for (unsigned rows : {16u, 24u, 32u}) {
        auto x = d.tensor_f32(uint64_t(rows) * cols, "wide add timing input");
        upload(d, *x, random_values(x->count, 905));
        auto out = d.tensor_f32(uint64_t(rows) * features, "wide add timing output");
        auto scratch = d.tensor_f32(out->count, "wide add timing scratch");
        const auto residual = random_values(out->count, 906);
        for (unsigned trial = 0; trial < 3; ++trial) for (unsigned step = 0; step < 2; ++step) {
            const bool wide = (trial + step) % 2;
            detail::Q8VerificationMath math(d, true, true, wide);
            upload(d, *out, residual);
            auto run = [&] { check(math.add(*out, *out, *scratch, *weight, *x, rows, cols, features)); };
            for (unsigned i = 0; i < 3; ++i) run();
            check(d.synchronize());
            const auto start = std::chrono::steady_clock::now();
            for (unsigned i = 0; i < 20; ++i) run();
            check(d.synchronize());
            const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count()/20;
            std::printf("Q8 wide add timing features=%u cols=%u rows=%u trial=%u wide=%d ms=%.6f scope=warm_operator\n",
                        features, cols, rows, trial, wide, ms);
        }
    }
}
void benchmark_swiglu(DeviceBackend &d) {
    constexpr unsigned rows = 17408, cols = 5120;
    auto gate = q8_weight(d, rows, cols, 751), up = q8_weight(d, rows, cols, 752);
    for (unsigned batch : {2u, 4u, 5u, 7u, 8u}) {
        auto x = d.tensor_f32(uint64_t(batch) * cols, "swiglu benchmark input");
        upload(d, *x, random_values(x->count, 753));
        auto out = d.tensor_f32(uint64_t(batch) * rows, "swiglu benchmark output");
        auto run = [&] { check(d.q8_0_matmul_silu_mul(*out, *gate, *up, *x, batch, cols, rows)); };
        setenv("QW3_Q8_SWIGLU_REUSE_ROWS", "0", 1);
        run();
        const auto expected = read(d, *out);
        for (const char *setting : {"1", "2", "4"}) {
            setenv("QW3_Q8_SWIGLU_REUSE_ROWS", setting, 1);
            run();
            compare(read(d, *out), expected, "SwiGLU reuse differs from reference");
        }
        const char *settings[] = {"0", "1", "2", "4"};
        for (unsigned trial = 0; trial < 3; ++trial) {
            for (unsigned step = 0; step < 4; ++step) {
                const char *setting = settings[(step + trial) % 4];
                setenv("QW3_Q8_SWIGLU_REUSE_ROWS", setting, 1);
                for (unsigned warm = 0; warm < 3; ++warm) run();
                check(d.synchronize());
                const auto start = std::chrono::steady_clock::now();
                for (unsigned repeat = 0; repeat < 20; ++repeat) run();
                check(d.synchronize());
                const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 20;
                std::printf("Q8 swiglu timing rows=%u cols=%u batch=%u trial=%u reuse_rows=%s ms=%.6f bit_exact=1 scope=warm_operator\n",
                            rows, cols, batch, trial, setting, ms);
                std::fflush(stdout);
            }
        }
    }
}
void recurrent(DeviceBackend &d, unsigned k, unsigned batch) {
    constexpr unsigned heads_k = 1, heads_v = 2, dim = 128;
    constexpr unsigned proj_width = (2 * heads_k + heads_v) * dim;
    constexpr unsigned core_width = heads_v * dim, state_width = heads_v * dim * dim;
    const unsigned conv_width = proj_width * (k - 1);
    auto proj_h = random_values(batch * proj_width, 123);
    auto gate_h = random_values(batch * core_width, 124);
    auto alpha_h = random_values(batch * heads_v, 125);
    auto beta_h = random_values(batch * heads_v, 126);
    auto state_h = random_values(state_width, 127, .01f), conv_h = random_values(conv_width, 128);
    auto cw_h = random_values(proj_width * k, 129);
    std::vector<float> a_h(heads_v, -.2f), dt_h(heads_v, .05f), norm_h(dim, 1.f);
    auto cw = d.weight_f32(cw_h.data(), cw_h.size(), "conv"), a = d.weight_f32(a_h.data(), a_h.size(), "a");
    auto dt = d.weight_f32(dt_h.data(), dt_h.size(), "dt"), norm = d.weight_f32(norm_h.data(), norm_h.size(), "norm");
    auto s = d.tensor_f32(state_width, "state"), cs = d.tensor_f32(conv_width, "conv state");
    auto proj = d.tensor_f32(proj_width, "proj"), gate = d.tensor_f32(core_width, "gate");
    auto alpha = d.tensor_f32(heads_v, "alpha"), beta = d.tensor_f32(heads_v, "beta");
    auto co = d.tensor_f32(proj_width, "conv out"), core = d.tensor_f32(core_width, "core");
    upload(d, *s, state_h); upload(d, *cs, conv_h);
    std::vector<float> expected_core, expected_conv, expected_s, expected_cs;
    auto append = [&](std::vector<float> &out, const DeviceTensor &t) {
        auto v = read(d, t); out.insert(out.end(), v.begin(), v.end());
    };
    for (unsigned b = 0; b < batch; ++b) {
        check(d.copy_bytes_from_host(*proj, 0, proj_h.data() + b * proj_width, proj_width * 4));
        check(d.copy_bytes_from_host(*gate, 0, gate_h.data() + b * core_width, core_width * 4));
        check(d.copy_bytes_from_host(*alpha, 0, alpha_h.data() + b * heads_v, heads_v * 4));
        check(d.copy_bytes_from_host(*beta, 0, beta_h.data() + b * heads_v, heads_v * 4));
        check(d.recurrent_single_token(*core, *s, *cs, *co, *proj, *gate, *alpha, *beta,
            *cw, *a, *dt, *norm, heads_k, heads_v, dim, dim, k, 1e-6f));
        append(expected_core, *core); append(expected_conv, *co);
        append(expected_s, *s); append(expected_cs, *cs);
    }
    auto bp = d.tensor_f32(proj_h.size(), "batch proj"), bg = d.tensor_f32(gate_h.size(), "batch gate");
    auto ba = d.tensor_f32(alpha_h.size(), "batch alpha"), bb = d.tensor_f32(beta_h.size(), "batch beta");
    auto bc = d.tensor_f32(expected_core.size(), "batch core"), bco = d.tensor_f32(expected_conv.size(), "batch conv");
    auto sc = d.tensor_f32(expected_s.size(), "state checkpoints"), csc = d.tensor_f32(expected_cs.size(), "conv checkpoints");
    for (bool checkpoints : {false, true}) {
        upload(d, *s, state_h); upload(d, *cs, conv_h);
        upload(d, *bp, proj_h); upload(d, *bg, gate_h); upload(d, *ba, alpha_h); upload(d, *bb, beta_h);
        check(d.recurrent_batch(*bc, *s, *cs, *bco, *bp, *bg, *ba, *bb, *cw, *a, *dt, *norm,
            batch, heads_k, heads_v, dim, dim, k, proj_width, proj_width, core_width,
            heads_v, heads_v, core_width, 1e-6f, checkpoints ? sc.get() : nullptr,
            checkpoints ? csc.get() : nullptr, checkpoints ? batch : 0));
        compare(read(d, *bco), expected_conv, "conv/normalization order differs");
        compare(read(d, *bc), expected_core, "recurrent core differs", 2e-6f);
        compare(read(d, *s), std::vector<float>(expected_s.end() - state_width, expected_s.end()), "final state differs", 2e-6f);
        compare(read(d, *cs), std::vector<float>(expected_cs.end() - conv_width, expected_cs.end()), "final conv state differs");
        if (checkpoints) {
            compare(read(d, *sc), expected_s, "state checkpoint differs", 2e-6f);
            compare(read(d, *csc), expected_cs, "conv checkpoint differs");
        }
        std::printf("recurrent k=%u batch=%u checkpoints=%d conv_exact=1 core/state=ok\n", k, batch, checkpoints);
    }
    // Treat each serial prefix as a separate request. This checks cross-request
    // kernels against the same independently executed single-token oracle.
    auto initial_s = expected_s, initial_cs = expected_cs;
    std::copy_backward(initial_s.begin(), initial_s.end() - state_width, initial_s.end());
    std::copy_backward(initial_cs.begin(), initial_cs.end() - conv_width, initial_cs.end());
    std::copy(state_h.begin(), state_h.end(), initial_s.begin());
    std::copy(conv_h.begin(), conv_h.end(), initial_cs.begin());
    upload(d, *sc, initial_s); upload(d, *csc, initial_cs);
    upload(d, *ba, alpha_h); upload(d, *bb, beta_h);
    check(d.recurrent_batch_independent(*bc, *sc, *csc, *bco, *bp, *bg, *ba, *bb,
        *cw, *a, *dt, *norm, batch, heads_k, heads_v, dim, dim, k, proj_width,
        proj_width, core_width, heads_v, heads_v, core_width, state_width, conv_width, 1e-6f));
    compare(read(d, *bco), expected_conv, "independent conv order differs");
    compare(read(d, *bc), expected_core, "independent core differs", 2e-6f);
    compare(read(d, *sc), expected_s, "independent state differs", 2e-6f);
    compare(read(d, *csc), expected_cs, "independent conv state differs");

    // Include an empty request and two differently sized token ranges.
    const unsigned split = batch / 2;
    std::vector<int32_t> indptr = {0, 0, static_cast<int32_t>(split), static_cast<int32_t>(batch)};
    std::vector<float> ragged_s, ragged_cs, final_s, final_cs;
    for (unsigned request = 0; request < 3; ++request) {
        for (bool end : {false, true}) {
            const unsigned offset = indptr[request + unsigned(end)];
            const float *rs = offset ? expected_s.data() + (offset - 1) * state_width : state_h.data();
            const float *rcs = offset ? expected_cs.data() + (offset - 1) * conv_width : conv_h.data();
            auto &sv = end ? final_s : ragged_s; auto &cv = end ? final_cs : ragged_cs;
            sv.insert(sv.end(), rs, rs + state_width); cv.insert(cv.end(), rcs, rcs + conv_width);
        }
    }
    auto rs = d.tensor_f32(ragged_s.size(), "ragged state"), rcs = d.tensor_f32(ragged_cs.size(), "ragged conv state");
    auto ptrs = d.tensor_i32(indptr.size(), "ragged indptr");
    check(d.copy_i32_from_host(*ptrs, 0, indptr.data(), indptr.size()));
    upload(d, *rs, ragged_s); upload(d, *rcs, ragged_cs);
    upload(d, *ba, alpha_h); upload(d, *bb, beta_h);
    check(d.recurrent_batch_ragged(*bc, *rs, *rcs, *bco, *bp, *bg, *ba, *bb, *ptrs, indptr.data(),
        *cw, *a, *dt, *norm, 3, batch, heads_k, heads_v, dim, dim, k, proj_width,
        proj_width, core_width, heads_v, heads_v, core_width, state_width, conv_width, 1e-6f));
    compare(read(d, *bco), expected_conv, "ragged conv order differs");
    compare(read(d, *bc), expected_core, "ragged core differs", 2e-6f);
    compare(read(d, *rs), final_s, "ragged state differs", 2e-6f);
    compare(read(d, *rcs), final_cs, "ragged conv state differs");
    std::printf("recurrent k=%u tokens=%u independent/ragged=ok\n", k, batch);
}
}
int main(int argc, char **argv) try {
    auto d = qw3::make_cuda_device_backend(qw3::LinearBackend::Auto); check(d->begin());
    if (argc == 2 && std::strcmp(argv[1], "--mma-small-only") == 0) {
        small_mma_dispatch(*d); check(d->end()); return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--mma-fanout-only") == 0) {
        if (!d->supports_q8_verify_mma()) throw std::runtime_error("MMA backend unavailable");
        for (unsigned batch : {1u, 8u, 9u, 16u, 24u, 32u, 33u, 64u, 65u}) {
            wide_fanout(*d, 5120, batch, 4, true, true);
            wide_fanout(*d, 6144, batch, 1, false, true);
            wide_fanout(*d, 17408, batch, 1, true, true);
        }
        check(d->end()); return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--wide-add-only") == 0) {
        wide_add(*d, 96, 37, false);
        wide_add(*d, 5120, 37, true);
        wide_add(*d, 6144, 5120, false);
        check(d->end()); return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--bench-wide-add") == 0) {
        benchmark_wide_add(*d); check(d->end()); return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--wide-swiglu-only") == 0) {
        wide_swiglu(*d, 96, 37, false);
        wide_swiglu(*d, 5120, 37, true);
        wide_swiglu(*d, 5120, 17408, false);
        check(d->end()); return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--bench-wide") == 0) {
        benchmark_wide(*d); check(d->end()); return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--bench-swiglu") == 0) {
        benchmark_swiglu(*d); check(d->end()); return 0;
    }
    const bool linear_only = argc == 2 && std::strcmp(argv[1], "--linear-only") == 0;
    if (argc != 1 && !linear_only) throw std::runtime_error("unknown Q8 verifier probe argument");
    // Odd scale-plane lengths and four-byte-only alignment exercise the
    // vector-load fallback as well as the aligned real model dimensions.
    for (unsigned cols : {32u, 64u, 96u, 256u, 5120u, 17408u})
        for (unsigned batch = 1; batch <= 8; ++batch) linear(*d, cols, batch);
    for (unsigned cols : {256u, 5120u, 17408u})
        for (unsigned batch : {9u, 16u, 31u, 33u, 49u, 65u, 128u})
            for (bool contiguous : {false, true}) linear(*d, cols, batch, true, contiguous);
    // Exercise the exact production shape, including padded residual guards:
    // the streaming-weight dispatch is deliberately restricted to FFN down.
    linear(*d, 17408, 8, false, false, 5120);
    // Eight-token aligned fanout path: include multiple outputs and padded input.
    for (unsigned count : {2u, 3u, 4u})
        for (bool padded : {false, true}) wide_fanout(*d, 5120, 8, count, padded);
    if (d->supports_q8_verify_fanout())
        for (unsigned cols : {256u, 5120u})
            for (unsigned batch : {9u, 16u, 31u, 33u, 49u, 65u, 128u})
                for (unsigned count : {2u, 3u, 4u})
                    for (bool padded : {false, true}) wide_fanout(*d, cols, batch, count, padded);
    if (!linear_only) for (unsigned k : {3u, 4u, 5u, 7u})
        for (unsigned batch : {1u, 2u, 4u, 7u, 8u, 65u}) recurrent(*d, k, batch);
    check(d->end()); return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "Q8 verifier kernel parity failed: %s\n", e.what()); return 1;
}
