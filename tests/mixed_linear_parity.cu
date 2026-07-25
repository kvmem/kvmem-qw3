#include "qw3/device_backend.hpp"
#include "qw3/model_source.hpp"
#include "qw3/tokenizer.hpp"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

void require(const qw3::DeviceStatus &status, const char *operation) {
    if (!status.ok) {
        std::fprintf(stderr, "%s failed: %s\n", operation, status.message);
        std::exit(1);
    }
}

void require_close(const std::vector<float> &got,
                   const std::vector<float> &expected,
                   float absolute_tolerance,
                   float relative_tolerance,
                   const char *name) {
    if (got.size() != expected.size()) {
        std::fprintf(stderr, "%s size mismatch\n", name);
        std::exit(1);
    }
    float max_absolute = 0.0f;
    float max_relative = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        const float absolute = std::fabs(got[i] - expected[i]);
        const float relative = absolute / std::max(1.0f, std::fabs(expected[i]));
        max_absolute = std::max(max_absolute, absolute);
        max_relative = std::max(max_relative, relative);
        if (absolute > absolute_tolerance && relative > relative_tolerance) {
            std::fprintf(stderr,
                         "%s mismatch at %zu: got=%g expected=%g abs=%g rel=%g\n",
                         name, i, got[i], expected[i], absolute, relative);
            std::exit(1);
        }
    }
    std::printf("%s: max_abs=%g max_rel=%g\n", name, max_absolute, max_relative);
}

std::vector<float> run_linear(qw3::DeviceBackend &backend,
                              const qw3::DeviceWeight &weight,
                              const std::vector<float> &input,
                              uint32_t batch,
                              uint32_t cols,
                              uint32_t rows) {
    auto x = backend.tensor_f32(static_cast<uint64_t>(batch) * cols, "mixed_test_x");
    auto out = backend.tensor_f32(static_cast<uint64_t>(batch) * rows, "mixed_test_out");
    require(backend.copy_bytes_from_host(
                *x, 0, input.data(), input.size() * sizeof(float)),
            "copy input");
    require(backend.linear_matmul(*out, weight, *x, batch, cols, rows),
            "linear matmul");
    std::vector<float> result(static_cast<size_t>(batch) * rows);
    require(backend.copy_to_host(*out, result.data(), 0, result.size()),
            "copy output");
    return result;
}

std::vector<float> copy_bf16_to_float(qw3::DeviceBackend &backend,
                                      const qw3::DeviceTensor &tensor) {
    std::vector<__nv_bfloat16> packed(tensor.count);
    require(backend.copy_bytes_to_host(
                tensor, packed.data(), 0,
                packed.size() * sizeof(__nv_bfloat16)),
            "copy BF16 tensor");
    std::vector<float> result(packed.size());
    for (size_t i = 0; i < packed.size(); ++i) {
        result[i] = __bfloat162float(packed[i]);
    }
    return result;
}

void test_bf16(qw3::DeviceBackend &backend) {
    constexpr uint32_t rows = 128;
    constexpr uint32_t cols = 128;
    constexpr uint32_t batch = 3;
    std::vector<__nv_bfloat16> weight(static_cast<size_t>(rows) * cols);
    std::vector<float> weight_value(weight.size());
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            const float value = static_cast<float>(static_cast<int>((row + col) % 11) - 5) / 8.0f;
            const size_t index = static_cast<size_t>(row) * cols + col;
            weight[index] = __float2bfloat16(value);
            weight_value[index] = __bfloat162float(weight[index]);
        }
    }
    std::vector<float> input(static_cast<size_t>(batch) * cols);
    std::vector<float> input_value(input.size());
    for (uint32_t item = 0; item < batch; ++item) {
        for (uint32_t col = 0; col < cols; ++col) {
            const float value = static_cast<float>(static_cast<int>((item * 3 + col) % 9) - 4) / 7.0f;
            const size_t index = static_cast<size_t>(item) * cols + col;
            const __nv_bfloat16 rounded = __float2bfloat16(value);
            input[index] = value;
            input_value[index] = __bfloat162float(rounded);
        }
    }
    auto device_weight = backend.weight_bf16(weight.data(), rows, cols, "linear.weight");
    const auto got = run_linear(backend, *device_weight, input, batch, cols, rows);
    std::vector<float> expected(got.size(), 0.0f);
    for (uint32_t item = 0; item < batch; ++item) {
        for (uint32_t row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (uint32_t col = 0; col < cols; ++col) {
                sum += weight_value[static_cast<size_t>(row) * cols + col] *
                       input_value[static_cast<size_t>(item) * cols + col];
            }
            expected[static_cast<size_t>(item) * rows + row] = sum;
        }
    }
    require_close(got, expected, 0.03f, 0.003f, "BF16 linear");
}

void test_fp8_shape(qw3::DeviceBackend &backend, uint32_t batch) {
    constexpr uint32_t rows = 128;
    constexpr uint32_t cols = 128;
    const __nv_fp8_e4m3 one(1.0f);
    std::vector<uint8_t> weight(static_cast<size_t>(rows) * cols, one.__x);
    std::vector<float> scales(rows);
    for (uint32_t row = 0; row < rows; ++row) {
        scales[row] = 0.5f + static_cast<float>(row) / 256.0f;
    }
    std::vector<float> input(static_cast<size_t>(batch) * cols);
    for (uint32_t item = 0; item < batch; ++item) {
        const float value = 0.5f * static_cast<float>(item + 1);
        std::fill(input.begin() + static_cast<size_t>(item) * cols,
                  input.begin() + static_cast<size_t>(item + 1) * cols, value);
    }
    auto device_weight = backend.weight_fp8_e4m3(
        weight.data(), scales.data(), scales.size(), rows, cols, "linear.weight");
    const auto got = run_linear(backend, *device_weight, input, batch, cols, rows);
    std::vector<float> expected(got.size());
    for (uint32_t item = 0; item < batch; ++item) {
        for (uint32_t row = 0; row < rows; ++row) {
            expected[static_cast<size_t>(item) * rows + row] =
                static_cast<float>(cols) * input[static_cast<size_t>(item) * cols] * scales[row];
        }
    }
    const char *name = batch <= 8 ? "FP8 decode linear" : "FP8 prefill linear";
    require_close(got, expected, 1.0f, 0.01f, name);
}

void test_fp8_bf16_prefill(qw3::DeviceBackend &backend) {
    constexpr uint32_t rows = 128;
    constexpr uint32_t cols = 128;
    constexpr uint32_t batch = 512;
    const __nv_fp8_e4m3 one(1.0f);
    std::vector<uint8_t> weight(
        static_cast<size_t>(rows) * cols, one.__x);
    std::vector<float> scales(rows);
    for (uint32_t row = 0; row < rows; ++row) {
        scales[row] = 0.5f + static_cast<float>(row % 16U) / 32.0f;
    }
    std::vector<float> input(static_cast<size_t>(batch) * cols);
    for (uint32_t item = 0; item < batch; ++item) {
        const float value =
            0.0625f * static_cast<float>(1U + item % 8U);
        std::fill(input.begin() + static_cast<size_t>(item) * cols,
                  input.begin() + static_cast<size_t>(item + 1) * cols,
                  value);
    }
    auto device_weight = backend.weight_fp8_e4m3(
        weight.data(), scales.data(), scales.size(),
        rows, cols, "fp8_bf16_prefill.weight");
    auto x = backend.tensor_f32(input.size(), "fp8_bf16_prefill.x");
    auto out = backend.tensor_bf16(
        static_cast<uint64_t>(batch) * rows, "fp8_bf16_prefill.out");
    require(backend.copy_bytes_from_host(
                *x, 0, input.data(), input.size() * sizeof(float)),
            "copy FP8 BF16 prefill input");
    require(backend.q8_0_matmul(
                *out, *device_weight, *x, batch, cols, rows),
            "FP8 BF16 prefill matmul");
    const auto got = copy_bf16_to_float(backend, *out);
    std::vector<float> expected(got.size());
    for (uint32_t item = 0; item < batch; ++item) {
        const float value = input[static_cast<size_t>(item) * cols];
        for (uint32_t row = 0; row < rows; ++row) {
            expected[static_cast<size_t>(item) * rows + row] =
                static_cast<float>(cols) * value * scales[row];
        }
    }
    require_close(
        got, expected, 0.5f, 0.005f, "FP8 BF16 prefill linear");
}

void test_fp8_fused_rms_fanout(qw3::DeviceBackend &backend) {
    constexpr uint32_t rows = 128;
    constexpr uint32_t cols = 128;
    constexpr uint32_t batch = 512;
    constexpr float eps = 1e-6f;
    std::vector<__nv_bfloat16> input(
        static_cast<size_t>(batch) * cols);
    for (uint32_t item = 0; item < batch; ++item) {
        for (uint32_t col = 0; col < cols; ++col) {
            input[static_cast<size_t>(item) * cols + col] =
                __float2bfloat16(
                    static_cast<float>(
                        static_cast<int>((item * 7 + col * 5) % 41) - 20) /
                    16.0f);
        }
    }
    std::vector<float> norm_data(cols);
    for (uint32_t col = 0; col < cols; ++col) {
        norm_data[col] =
            0.75f + static_cast<float>(col % 13U) / 32.0f;
    }
    std::vector<uint8_t> weight_data[3];
    std::vector<float> weight_scales[3];
    for (uint32_t slot = 0; slot < 3; ++slot) {
        weight_data[slot].resize(static_cast<size_t>(rows) * cols);
        weight_scales[slot].resize(rows);
        for (uint32_t row = 0; row < rows; ++row) {
            weight_scales[slot][row] =
                0.25f + static_cast<float>((row + slot * 3U) % 17U) /
                            32.0f;
            for (uint32_t col = 0; col < cols; ++col) {
                const float value =
                    static_cast<float>(
                        static_cast<int>((row + col + slot * 5U) % 7U) - 3) /
                    2.0f;
                weight_data[slot][static_cast<size_t>(row) * cols + col] =
                    __nv_fp8_e4m3(value).__x;
            }
        }
    }

    auto x = backend.tensor_bf16(input.size(), "fused_rms_fanout.x");
    auto normalized = backend.tensor_f32(
        input.size(), "fused_rms_fanout.normalized");
    auto fused_scratch = backend.tensor_f32(
        input.size(), "fused_rms_fanout.fused_scratch");
    auto norm = backend.weight_f32(
        norm_data.data(), norm_data.size(), "fused_rms_fanout.norm.weight");
    require(backend.copy_bytes_from_host(
                *x, 0, input.data(),
                input.size() * sizeof(__nv_bfloat16)),
            "copy fused RMS fanout input");

    std::unique_ptr<qw3::DeviceWeight> weights[3];
    std::unique_ptr<qw3::DeviceTensor> reference[3];
    std::unique_ptr<qw3::DeviceTensor> fused[3];
    const qw3::DeviceWeight *weight_ptrs[3] = {};
    qw3::DeviceTensor *reference_ptrs[3] = {};
    qw3::DeviceTensor *fused_ptrs[3] = {};
    const uint32_t strides[3] = {rows, rows, rows};
    for (uint32_t slot = 0; slot < 3; ++slot) {
        weights[slot] = backend.weight_fp8_e4m3(
            weight_data[slot].data(), weight_scales[slot].data(), rows,
            rows, cols, "fused_rms_fanout.linear.weight");
        reference[slot] = backend.tensor_f32(
            static_cast<uint64_t>(batch) * rows,
            "fused_rms_fanout.reference");
        fused[slot] = backend.tensor_f32(
            static_cast<uint64_t>(batch) * rows,
            "fused_rms_fanout.output");
        weight_ptrs[slot] = weights[slot].get();
        reference_ptrs[slot] = reference[slot].get();
        fused_ptrs[slot] = fused[slot].get();
    }

    require(backend.rms_norm_batch(
                *normalized, *x, *norm, batch, cols, eps),
            "reference RMSNorm for fused fanout");
    require(backend.q8_0_matmul_fanout(
                reference_ptrs, weight_ptrs, strides, 3,
                *normalized, batch, cols),
            "reference FP8 fanout after RMSNorm");
    require(backend.rms_norm_q8_0_matmul_fanout(
                *fused_scratch, fused_ptrs, weight_ptrs, strides, 3,
                *x, *norm, batch, cols, eps),
            "fused RMSNorm FP8 fanout");

    for (uint32_t slot = 0; slot < 3; ++slot) {
        std::vector<float> got(static_cast<size_t>(batch) * rows);
        std::vector<float> expected(got.size());
        require(backend.copy_to_host(
                    *fused[slot], got.data(), 0, got.size()),
                "copy fused RMS fanout output");
        require(backend.copy_to_host(
                    *reference[slot], expected.data(), 0, expected.size()),
                "copy reference RMS fanout output");
        const std::string name =
            "FP8 fused RMS fanout " + std::to_string(slot);
        require_close(got, expected, 0.01f, 0.0001f, name.c_str());
    }

    std::vector<__nv_bfloat16> bf16_weight_data[2];
    std::unique_ptr<qw3::DeviceWeight> bf16_weights[2];
    for (uint32_t slot = 0; slot < 2; ++slot) {
        bf16_weight_data[slot].resize(static_cast<size_t>(rows) * cols);
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t col = 0; col < cols; ++col) {
                bf16_weight_data[slot][
                    static_cast<size_t>(row) * cols + col] =
                    __float2bfloat16(
                        static_cast<float>(
                            static_cast<int>(
                                (row * 3 + col + slot * 7U) % 17U) -
                            8) /
                        16.0f);
            }
        }
        bf16_weights[slot] = backend.weight_bf16(
            bf16_weight_data[slot].data(), rows, cols,
            "fused_rms_fanout.bf16.weight");
    }
    std::unique_ptr<qw3::DeviceTensor> mixed_reference[4];
    std::unique_ptr<qw3::DeviceTensor> mixed_fused[4];
    qw3::DeviceTensor *mixed_reference_ptrs[4] = {};
    qw3::DeviceTensor *mixed_fused_ptrs[4] = {};
    const qw3::DeviceWeight *mixed_weight_ptrs[4] = {
        weights[0].get(), weights[1].get(),
        bf16_weights[0].get(), bf16_weights[1].get(),
    };
    const uint32_t mixed_strides[4] = {rows, rows, rows, rows};
    for (uint32_t slot = 0; slot < 4; ++slot) {
        mixed_reference[slot] = backend.tensor_f32(
            static_cast<uint64_t>(batch) * rows,
            "fused_rms_mixed.reference");
        mixed_fused[slot] = backend.tensor_f32(
            static_cast<uint64_t>(batch) * rows,
            "fused_rms_mixed.output");
        mixed_reference_ptrs[slot] = mixed_reference[slot].get();
        mixed_fused_ptrs[slot] = mixed_fused[slot].get();
    }
    require(backend.q8_0_matmul_fanout(
                mixed_reference_ptrs, mixed_weight_ptrs, mixed_strides, 4,
                *normalized, batch, cols),
            "reference mixed fanout after RMSNorm");
    require(backend.rms_norm_q8_0_matmul_fanout(
                *fused_scratch, mixed_fused_ptrs, mixed_weight_ptrs,
                mixed_strides, 4, *x, *norm, batch, cols, eps),
            "fused RMSNorm mixed fanout");
    for (uint32_t slot = 0; slot < 4; ++slot) {
        std::vector<float> got(static_cast<size_t>(batch) * rows);
        std::vector<float> expected(got.size());
        require(backend.copy_to_host(
                    *mixed_fused[slot], got.data(), 0, got.size()),
                "copy fused RMS mixed fanout output");
        require(backend.copy_to_host(
                    *mixed_reference[slot], expected.data(), 0,
                    expected.size()),
                "copy reference RMS mixed fanout output");
        const std::string name =
            "FP8/BF16 fused RMS fanout " + std::to_string(slot);
        require_close(got, expected, 0.01f, 0.0001f, name.c_str());
    }
}

void test_fp8_fused_paths(qw3::DeviceBackend &backend) {
    const char *enabled = std::getenv("QW3_FP8_SMALL_CUBLASLT");
    if (enabled && (std::string(enabled) == "0" ||
                    std::string(enabled) == "off" ||
                    std::string(enabled) == "false")) {
        return;
    }
    constexpr uint32_t rows = 128;
    constexpr uint32_t cols = 128;
    const __nv_fp8_e4m3 one(1.0f);
    const __nv_fp8_e4m3 negative_one(-1.0f);
    std::vector<uint8_t> positive(
        static_cast<size_t>(rows) * cols, one.__x);
    std::vector<uint8_t> negative(
        static_cast<size_t>(rows) * cols, negative_one.__x);
    std::vector<float> scales(rows);
    for (uint32_t row = 0; row < rows; ++row) {
        scales[row] = 0.5f + static_cast<float>(row % 8U) / 16.0f;
    }
    auto positive_device = backend.weight_fp8_e4m3(
        positive.data(), scales.data(), scales.size(),
        rows, cols, "fp8.positive");
    auto negative_device = backend.weight_fp8_e4m3(
        negative.data(), scales.data(), scales.size(),
        rows, cols, "fp8.negative");
    auto third_device = backend.weight_fp8_e4m3(
        positive.data(), scales.data(), scales.size(),
        rows, cols, "fp8.third");
    std::vector<__nv_bfloat16> bf16(
        static_cast<size_t>(rows) * cols, __float2bfloat16(0.25f));
    std::vector<__nv_bfloat16> bf16_alternate(
        static_cast<size_t>(rows) * cols);
    for (size_t i = 0; i < bf16_alternate.size(); ++i) {
        bf16_alternate[i] = __float2bfloat16(
            static_cast<float>(static_cast<int>(i % 17U) - 8) / 32.0f);
    }
    auto bf16_device = backend.weight_bf16(
        bf16.data(), rows, cols, "bf16.positive");
    auto bf16_alternate_device = backend.weight_bf16(
        bf16_alternate.data(), rows, cols, "bf16.alternate");
    require(backend.prepare_fp8_fanout_pair(
                *positive_device, *negative_device),
            "prepare FP8 parity pair");

    std::vector<float> input(cols, 0.5f);
    auto x = backend.tensor_f32(cols, "fp8_fused_x");
    require(backend.copy_bytes_from_host(
                *x, 0, input.data(), input.size() * sizeof(float)),
            "copy FP8 fused input");

    auto out0 = backend.tensor_f32(rows, "fp8_fanout_0");
    auto out1 = backend.tensor_f32(rows, "fp8_fanout_1");
    qw3::DeviceTensor *outs[2] = {out0.get(), out1.get()};
    const qw3::DeviceWeight *weights[2] = {
        positive_device.get(), negative_device.get()};
    require(backend.q8_0_matvec_fanout(outs, weights, 2, *x),
            "FP8 fanout");
    std::vector<float> got0(rows);
    std::vector<float> got1(rows);
    require(backend.copy_to_host(*out0, got0.data(), 0, got0.size()),
            "copy FP8 fanout 0");
    require(backend.copy_to_host(*out1, got1.data(), 0, got1.size()),
            "copy FP8 fanout 1");
    std::vector<float> expected0(rows);
    std::vector<float> expected1(rows);
    for (uint32_t row = 0; row < rows; ++row) {
        expected0[row] = 0.5f * cols * scales[row];
        expected1[row] = -expected0[row];
    }
    require_close(got0, expected0, 0.1f, 0.001f, "FP8 fused fanout 0");
    require_close(got1, expected1, 0.1f, 0.001f, "FP8 fused fanout 1");

    auto mixed_out0 = backend.tensor_f32(rows, "fp8_mixed_fanout_0");
    auto mixed_out1 = backend.tensor_f32(rows, "fp8_mixed_fanout_1");
    auto mixed_out2 = backend.tensor_f32(rows, "bf16_mixed_fanout_2");
    auto mixed_out3 = backend.tensor_f32(rows, "bf16_mixed_fanout_3");
    qw3::DeviceTensor *mixed_outs[4] = {
        mixed_out0.get(), mixed_out1.get(),
        mixed_out2.get(), mixed_out3.get()};
    const qw3::DeviceWeight *mixed_weights[4] = {
        positive_device.get(), negative_device.get(),
        bf16_device.get(), bf16_alternate_device.get()};
    require(backend.q8_0_matvec_fanout(
                mixed_outs, mixed_weights, 4, *x),
            "mixed FP8/BF16 fanout");
    std::vector<float> got_mixed0(rows);
    std::vector<float> got_mixed1(rows);
    std::vector<float> got_mixed2(rows);
    std::vector<float> got_mixed3(rows);
    require(backend.copy_to_host(
                *mixed_out0, got_mixed0.data(), 0, got_mixed0.size()),
            "copy mixed FP8 fanout 0");
    require(backend.copy_to_host(
                *mixed_out1, got_mixed1.data(), 0, got_mixed1.size()),
            "copy mixed FP8 fanout 1");
    require(backend.copy_to_host(
                *mixed_out2, got_mixed2.data(), 0, got_mixed2.size()),
            "copy mixed BF16 fanout 2");
    require(backend.copy_to_host(
                *mixed_out3, got_mixed3.data(), 0, got_mixed3.size()),
            "copy mixed BF16 fanout 3");
    std::vector<float> expected_mixed2(rows, 0.25f * 0.5f * cols);
    std::vector<float> expected_mixed3(rows);
    for (uint32_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0; col < cols; ++col) {
            sum += __bfloat162float(
                       bf16_alternate[static_cast<size_t>(row) * cols + col]) *
                   0.5f;
        }
        expected_mixed3[row] = sum;
    }
    require_close(got_mixed0, expected0, 0.1f, 0.001f,
                  "mixed FP8 fanout 0");
    require_close(got_mixed1, expected1, 0.1f, 0.001f,
                  "mixed FP8 fanout 1");
    require_close(got_mixed2, expected_mixed2, 0.1f, 0.001f,
                  "mixed BF16 fanout 2");
    require_close(got_mixed3, expected_mixed3, 0.1f, 0.001f,
                  "mixed BF16 fanout 3");

    constexpr uint32_t batch = 4;
    std::vector<float> batch_input(static_cast<size_t>(batch) * cols);
    for (uint32_t item = 0; item < batch; ++item) {
        std::fill(
            batch_input.begin() + static_cast<size_t>(item) * cols,
            batch_input.begin() + static_cast<size_t>(item + 1) * cols,
            0.25f * static_cast<float>(item + 1));
    }
    auto batch_x = backend.tensor_f32(
        batch_input.size(), "fp8_fused_batch_x");
    require(backend.copy_bytes_from_host(
                *batch_x, 0, batch_input.data(),
                batch_input.size() * sizeof(float)),
            "copy FP8 fused batch input");
    auto batch_out0 = backend.tensor_f32(
        static_cast<uint64_t>(batch) * rows, "fp8_batch_fanout_0");
    auto batch_out1 = backend.tensor_f32(
        static_cast<uint64_t>(batch) * rows, "fp8_batch_fanout_1");
    auto batch_out2 = backend.tensor_f32(
        static_cast<uint64_t>(batch) * rows, "bf16_batch_fanout_2");
    auto batch_out3 = backend.tensor_f32(
        static_cast<uint64_t>(batch) * rows, "bf16_batch_fanout_3");
    qw3::DeviceTensor *batch_outs[4] = {
        batch_out0.get(), batch_out1.get(),
        batch_out2.get(), batch_out3.get()};
    const uint32_t batch_strides[4] = {rows, rows, rows, rows};
    require(backend.q8_0_matmul_fanout(
                batch_outs, mixed_weights, batch_strides, 4,
                *batch_x, batch, cols),
            "mixed FP8/BF16 batch fanout");
    std::vector<float> got_batch0(static_cast<size_t>(batch) * rows);
    std::vector<float> got_batch1(static_cast<size_t>(batch) * rows);
    std::vector<float> got_batch2(static_cast<size_t>(batch) * rows);
    std::vector<float> got_batch3(static_cast<size_t>(batch) * rows);
    require(backend.copy_to_host(
                *batch_out0, got_batch0.data(), 0, got_batch0.size()),
            "copy batch FP8 fanout 0");
    require(backend.copy_to_host(
                *batch_out1, got_batch1.data(), 0, got_batch1.size()),
            "copy batch FP8 fanout 1");
    require(backend.copy_to_host(
                *batch_out2, got_batch2.data(), 0, got_batch2.size()),
            "copy batch BF16 fanout 2");
    require(backend.copy_to_host(
                *batch_out3, got_batch3.data(), 0, got_batch3.size()),
            "copy batch BF16 fanout 3");
    std::vector<float> expected_batch0(got_batch0.size());
    std::vector<float> expected_batch1(got_batch1.size());
    std::vector<float> expected_batch2(got_batch2.size());
    std::vector<float> expected_batch3(got_batch3.size());
    for (uint32_t item = 0; item < batch; ++item) {
        const float value = 0.25f * static_cast<float>(item + 1);
        for (uint32_t row = 0; row < rows; ++row) {
            const size_t index = static_cast<size_t>(item) * rows + row;
            expected_batch0[index] = value * cols * scales[row];
            expected_batch1[index] = -expected_batch0[index];
            expected_batch2[index] = value * cols * 0.25f;
            float sum = 0.0f;
            for (uint32_t col = 0; col < cols; ++col) {
                sum += __bfloat162float(
                           bf16_alternate[
                               static_cast<size_t>(row) * cols + col]) *
                       value;
            }
            expected_batch3[index] = sum;
        }
    }
    require_close(got_batch0, expected_batch0, 0.2f, 0.002f,
                  "mixed FP8 batch fanout 0");
    require_close(got_batch1, expected_batch1, 0.2f, 0.002f,
                  "mixed FP8 batch fanout 1");
    require_close(got_batch2, expected_batch2, 0.2f, 0.002f,
                  "mixed BF16 batch fanout 2");
    require_close(got_batch3, expected_batch3, 0.2f, 0.002f,
                  "mixed BF16 batch fanout 3");

    auto triple_out0 = backend.tensor_f32(
        static_cast<uint64_t>(batch) * rows, "fp8_triple_fanout_0");
    auto triple_out1 = backend.tensor_f32(
        static_cast<uint64_t>(batch) * rows, "fp8_triple_fanout_1");
    auto triple_out2 = backend.tensor_f32(
        static_cast<uint64_t>(batch) * rows, "fp8_triple_fanout_2");
    qw3::DeviceTensor *triple_outs[3] = {
        triple_out0.get(), triple_out1.get(), triple_out2.get()};
    const qw3::DeviceWeight *triple_weights[3] = {
        third_device.get(), positive_device.get(), negative_device.get()};
    const uint32_t triple_strides[3] = {rows, rows, rows};
    require(backend.q8_0_matmul_fanout(
                triple_outs, triple_weights, triple_strides, 3,
                *batch_x, batch, cols),
            "FP8 triple batch fanout");
    std::vector<float> got_triple0(got_batch0.size());
    std::vector<float> got_triple1(got_batch0.size());
    std::vector<float> got_triple2(got_batch0.size());
    require(backend.copy_to_host(
                *triple_out0, got_triple0.data(), 0, got_triple0.size()),
            "copy triple FP8 fanout 0");
    require(backend.copy_to_host(
                *triple_out1, got_triple1.data(), 0, got_triple1.size()),
            "copy triple FP8 fanout 1");
    require(backend.copy_to_host(
                *triple_out2, got_triple2.data(), 0, got_triple2.size()),
            "copy triple FP8 fanout 2");
    require_close(got_triple0, expected_batch0, 0.2f, 0.002f,
                  "FP8 triple batch fanout 0");
    require_close(got_triple1, expected_batch0, 0.2f, 0.002f,
                  "FP8 triple batch fanout 1");
    require_close(got_triple2, expected_batch1, 0.2f, 0.002f,
                  "FP8 triple batch fanout 2");

    auto batch_swiglu = backend.tensor_f32(
        static_cast<uint64_t>(batch) * rows, "fp8_batch_swiglu");
    require(backend.q8_0_matmul_silu_mul(
                *batch_swiglu, *positive_device, *negative_device,
                *batch_x, batch, cols, rows),
            "FP8 fused batch SwiGLU");
    std::vector<float> got_batch_swiglu(
        static_cast<size_t>(batch) * rows);
    require(backend.copy_to_host(
                *batch_swiglu, got_batch_swiglu.data(), 0,
                got_batch_swiglu.size()),
            "copy FP8 batch SwiGLU");
    std::vector<float> expected_batch_swiglu(got_batch_swiglu.size());
    for (size_t i = 0; i < expected_batch_swiglu.size(); ++i) {
        const float gate = expected_batch0[i];
        expected_batch_swiglu[i] =
            (gate / (1.0f + std::exp(-gate))) * expected_batch1[i];
    }
    require_close(
        got_batch_swiglu, expected_batch_swiglu, 2.0f, 0.002f,
        "FP8 fused batch SwiGLU");

    auto swiglu = backend.tensor_f32(rows, "fp8_swiglu");
    require(backend.q8_0_matvec_silu_mul(
                *swiglu, *positive_device, *positive_device, *x),
            "FP8 fused SwiGLU");
    std::vector<float> got_swiglu(rows);
    require(backend.copy_to_host(
                *swiglu, got_swiglu.data(), 0, got_swiglu.size()),
            "copy FP8 SwiGLU");
    std::vector<float> expected_swiglu(rows);
    for (uint32_t row = 0; row < rows; ++row) {
        const float projection = expected0[row];
        expected_swiglu[row] =
            (projection / (1.0f + std::exp(-projection))) * projection;
    }
    require_close(got_swiglu, expected_swiglu, 1.0f, 0.001f,
                  "FP8 fused SwiGLU");

    auto residual = backend.tensor_f32(rows, "fp8_residual");
    std::vector<float> residual_host(rows, 2.0f);
    require(backend.copy_bytes_from_host(
                *residual, 0, residual_host.data(),
                residual_host.size() * sizeof(float)),
            "copy FP8 residual");
    require(backend.q8_0_matvec_add(
                *residual, *positive_device, *x),
            "FP8 fused residual add");
    std::vector<float> got_add(rows);
    require(backend.copy_to_host(
                *residual, got_add.data(), 0, got_add.size()),
            "copy FP8 residual result");
    std::vector<float> expected_add(rows);
    for (uint32_t row = 0; row < rows; ++row) {
        expected_add[row] = 2.0f + expected0[row];
    }
    require_close(got_add, expected_add, 0.1f, 0.001f,
                  "FP8 fused residual add");

    auto bf16_batch_out = backend.tensor_bf16(
        static_cast<uint64_t>(batch) * rows, "fp8_bf16_batch_out");
    require(backend.q8_0_matmul(
                *bf16_batch_out, *positive_device, *batch_x,
                batch, cols, rows),
            "FP8 BF16 batch output");
    const auto got_bf16_batch =
        copy_bf16_to_float(backend, *bf16_batch_out);
    require_close(got_bf16_batch, expected_batch0, 0.5f, 0.005f,
                  "FP8 BF16 batch output");

    auto bf16_residual = backend.tensor_bf16(rows, "fp8_bf16_residual");
    std::vector<__nv_bfloat16> bf16_residual_host(
        rows, __float2bfloat16(2.0f));
    require(backend.copy_bytes_from_host(
                *bf16_residual, 0, bf16_residual_host.data(),
                bf16_residual_host.size() * sizeof(__nv_bfloat16)),
            "copy FP8 BF16 residual");
    require(backend.q8_0_matvec_add(
                *bf16_residual, *positive_device, *x),
            "FP8 BF16 fused residual add");
    const auto got_bf16_add =
        copy_bf16_to_float(backend, *bf16_residual);
    require_close(got_bf16_add, expected_add, 0.5f, 0.005f,
                  "FP8 BF16 fused residual add");

    std::vector<__nv_bfloat16> rms_input_host(
        static_cast<size_t>(batch) * cols);
    std::vector<float> rms_input_rounded(rms_input_host.size());
    for (uint32_t item = 0; item < batch; ++item) {
        for (uint32_t col = 0; col < cols; ++col) {
            const size_t index = static_cast<size_t>(item) * cols + col;
            rms_input_host[index] = __float2bfloat16(
                static_cast<float>(
                    static_cast<int>((item * 5 + col * 3) % 29) - 14) /
                16.0f);
            rms_input_rounded[index] =
                __bfloat162float(rms_input_host[index]);
        }
    }
    std::vector<float> rms_weight_host(cols);
    for (uint32_t col = 0; col < cols; ++col) {
        rms_weight_host[col] =
            0.75f + static_cast<float>(col % 11) / 32.0f;
    }
    auto rms_input = backend.tensor_bf16(
        rms_input_host.size(), "bf16_main_rms_input");
    auto rms_output = backend.tensor_f32(
        rms_input_host.size(), "bf16_main_rms_output");
    auto rms_weight = backend.weight_f32(
        rms_weight_host.data(), rms_weight_host.size(),
        "bf16_main_rms_weight");
    require(backend.copy_bytes_from_host(
                *rms_input, 0, rms_input_host.data(),
                rms_input_host.size() * sizeof(__nv_bfloat16)),
            "copy BF16 RMS input");
    constexpr float rms_eps = 1e-6f;
    require(backend.rms_norm_batch(
                *rms_output, *rms_input, *rms_weight,
                batch, cols, rms_eps),
            "BF16 main RMSNorm");
    std::vector<float> got_rms(rms_input_rounded.size());
    require(backend.copy_to_host(
                *rms_output, got_rms.data(), 0, got_rms.size()),
            "copy BF16 main RMSNorm");
    std::vector<float> expected_rms(got_rms.size());
    for (uint32_t item = 0; item < batch; ++item) {
        float sum_sq = 0.0f;
        for (uint32_t col = 0; col < cols; ++col) {
            const float value =
                rms_input_rounded[static_cast<size_t>(item) * cols + col];
            sum_sq += value * value;
        }
        const float inv =
            1.0f / std::sqrt(sum_sq / static_cast<float>(cols) + rms_eps);
        for (uint32_t col = 0; col < cols; ++col) {
            const size_t index = static_cast<size_t>(item) * cols + col;
            expected_rms[index] =
                rms_input_rounded[index] * inv * rms_weight_host[col];
        }
    }
    require_close(got_rms, expected_rms, 0.001f, 0.001f,
                  "BF16 main RMSNorm");
}

void test_nvfp4_shape(qw3::DeviceBackend &backend, uint32_t batch) {
    constexpr uint32_t rows = 128;
    constexpr uint32_t cols = 128;
    const uint8_t positive = __nv_cvt_float2_to_fp4x2(
        make_float2(1.0f, 1.0f), __NV_E2M1, cudaRoundNearest);
    const uint8_t negative = __nv_cvt_float2_to_fp4x2(
        make_float2(-1.0f, -1.0f), __NV_E2M1, cudaRoundNearest);
    std::vector<uint8_t> weight(static_cast<size_t>(rows) * cols / 2);
    for (uint32_t row = 0; row < rows; ++row) {
        std::fill(weight.begin() + static_cast<size_t>(row) * cols / 2,
                  weight.begin() + static_cast<size_t>(row + 1) * cols / 2,
                  row % 2 == 0 ? positive : negative);
    }
    const __nv_fp8_e4m3 one(1.0f);
    std::vector<uint8_t> scales(static_cast<size_t>(rows) * (cols / 16), one.__x);
    std::vector<float> input(static_cast<size_t>(batch) * cols, 6.0f);
    auto device_weight = backend.weight_nvfp4_e2m1(
        weight.data(), scales.data(), rows, cols / 16,
        1.0f, 1.0f, rows, cols, "linear.weight");
    const auto got = run_linear(backend, *device_weight, input, batch, cols, rows);
    std::vector<float> expected(got.size());
    for (uint32_t item = 0; item < batch; ++item) {
        for (uint32_t row = 0; row < rows; ++row) {
            expected[static_cast<size_t>(item) * rows + row] =
                (row % 2 == 0 ? 1.0f : -1.0f) * 6.0f * cols;
        }
    }
    const char *name = batch == 1 ? "NVFP4 decode linear" : "NVFP4 prefill linear";
    require_close(got, expected, 2.0f, 0.01f, name);
}

void test_nvfp4_fused_paths(qw3::DeviceBackend &backend) {
    constexpr uint32_t rows = 128;
    constexpr uint32_t cols = 128;
    constexpr uint32_t batch = 4;
    const uint8_t positive = __nv_cvt_float2_to_fp4x2(
        make_float2(1.0f, 1.0f), __NV_E2M1, cudaRoundNearest);
    const uint8_t negative = __nv_cvt_float2_to_fp4x2(
        make_float2(-1.0f, -1.0f), __NV_E2M1, cudaRoundNearest);
    std::vector<uint8_t> weight_positive(
        static_cast<size_t>(rows) * cols / 2, positive);
    std::vector<uint8_t> weight_negative(
        static_cast<size_t>(rows) * cols / 2, negative);
    const __nv_fp8_e4m3 one(1.0f);
    std::vector<uint8_t> scales(
        static_cast<size_t>(rows) * (cols / 16), one.__x);
    auto positive_device = backend.weight_nvfp4_e2m1(
        weight_positive.data(), scales.data(), rows, cols / 16,
        1.0f, 1.0f, rows, cols, "positive.weight");
    auto negative_device = backend.weight_nvfp4_e2m1(
        weight_negative.data(), scales.data(), rows, cols / 16,
        1.0f, 1.0f, rows, cols, "negative.weight");
    require(backend.prepare_nvfp4_fanout_pair(
                *positive_device, *negative_device),
            "prepare NVFP4 gate/up pair");

    std::vector<float> input(static_cast<size_t>(batch) * cols, 6.0f);
    auto x = backend.tensor_f32(input.size(), "nvfp4_fused_x");
    require(backend.copy_bytes_from_host(
                *x, 0, input.data(), input.size() * sizeof(float)),
            "copy fused input");

    auto out0 = backend.tensor_f32(static_cast<uint64_t>(batch) * rows,
                                   "nvfp4_fanout_0");
    auto out1 = backend.tensor_f32(static_cast<uint64_t>(batch) * rows,
                                   "nvfp4_fanout_1");
    qw3::DeviceTensor *outs[2] = {out0.get(), out1.get()};
    const qw3::DeviceWeight *weights[2] = {
        positive_device.get(), negative_device.get()};
    const uint32_t strides[2] = {rows, rows};
    require(backend.q8_0_matmul_fanout(
                outs, weights, strides, 2, *x, batch, cols),
            "NVFP4 pair fanout");
    std::vector<float> got0(static_cast<size_t>(batch) * rows);
    std::vector<float> got1(static_cast<size_t>(batch) * rows);
    require(backend.copy_to_host(*out0, got0.data(), 0, got0.size()),
            "copy fanout 0");
    require(backend.copy_to_host(*out1, got1.data(), 0, got1.size()),
            "copy fanout 1");
    std::vector<float> expected0(got0.size(), 6.0f * cols);
    std::vector<float> expected1(got1.size(), -6.0f * cols);
    require_close(got0, expected0, 2.0f, 0.01f, "NVFP4 pair fanout 0");
    require_close(got1, expected1, 2.0f, 0.01f, "NVFP4 pair fanout 1");

    auto residual = backend.tensor_f32(static_cast<uint64_t>(batch) * rows,
                                       "nvfp4_residual");
    auto tmp = backend.tensor_f32(static_cast<uint64_t>(batch) * rows,
                                  "nvfp4_add_tmp");
    std::vector<float> residual_host(static_cast<size_t>(batch) * rows, 2.0f);
    require(backend.copy_bytes_from_host(
                *residual, 0, residual_host.data(),
                residual_host.size() * sizeof(float)),
            "copy residual");
    require(backend.q8_0_matmul_add(
                *residual, *residual, *tmp, *positive_device, *x,
                batch, cols, rows),
            "NVFP4 batched residual add");
    std::vector<float> got_add(residual_host.size());
    require(backend.copy_to_host(
                *residual, got_add.data(), 0, got_add.size()),
            "copy residual result");
    std::vector<float> expected_add(got_add.size(), 2.0f + 6.0f * cols);
    require_close(got_add, expected_add, 2.0f, 0.01f,
                  "NVFP4 batched residual add");

    auto single_x = backend.tensor_f32(cols, "nvfp4_swiglu_x");
    require(backend.copy_bytes_from_host(
                *single_x, 0, input.data(), cols * sizeof(float)),
            "copy SwiGLU input");
    auto swiglu = backend.tensor_f32(rows, "nvfp4_swiglu");
    require(backend.q8_0_matvec_silu_mul(
                *swiglu, *positive_device, *negative_device, *single_x),
            "NVFP4 fused SwiGLU");
    std::vector<float> got_swiglu(rows);
    require(backend.copy_to_host(
                *swiglu, got_swiglu.data(), 0, got_swiglu.size()),
            "copy SwiGLU result");
    const float projection = 6.0f * cols;
    const float expected_swiglu_value =
        (projection / (1.0f + std::exp(-projection))) * -projection;
    std::vector<float> expected_swiglu(rows, expected_swiglu_value);
    require_close(got_swiglu, expected_swiglu, 8.0f, 0.001f,
                  "NVFP4 fused SwiGLU");

    constexpr uint32_t prefill_batch = 128;
    auto down_device = backend.weight_nvfp4_e2m1(
        weight_positive.data(), scales.data(), rows, cols / 16,
        1.0f, 1.0f, rows, cols, "down.weight");
    std::vector<float> norm_weight_host(cols);
    for (uint32_t col = 0; col < cols; ++col) {
        norm_weight_host[col] =
            0.75f + static_cast<float>(col % 9) / 32.0f;
    }
    auto norm_weight = backend.weight_f32(
        norm_weight_host.data(), norm_weight_host.size(), "ffn_norm.weight");
    std::vector<float> prefill_input(
        static_cast<size_t>(prefill_batch) * cols);
    for (uint32_t row = 0; row < prefill_batch; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            prefill_input[static_cast<size_t>(row) * cols + col] =
                static_cast<float>(
                    static_cast<int>((row * 7 + col * 3) % 31) - 15) /
                16.0f;
        }
    }
    auto prefill_x = backend.tensor_f32(
        prefill_input.size(), "nvfp4_ffn_prefill_x");
    require(backend.copy_bytes_from_host(
                *prefill_x, 0, prefill_input.data(),
                prefill_input.size() * sizeof(float)),
            "copy NVFP4 FFN prefill input");
    auto reference_mid = backend.tensor_f32(
        static_cast<uint64_t>(prefill_batch) * rows,
        "nvfp4_ffn_reference_mid");
    auto reference_norm = backend.tensor_f32(
        static_cast<uint64_t>(prefill_batch) * cols,
        "nvfp4_ffn_reference_norm");
    auto reference_out = backend.tensor_f32(
        static_cast<uint64_t>(prefill_batch) * rows,
        "nvfp4_ffn_reference_out");
    constexpr float norm_eps = 1e-6f;
    require(backend.rms_norm_batch(
                *reference_norm, *prefill_x, *norm_weight,
                prefill_batch, cols, norm_eps),
            "NVFP4 reference prefill RMSNorm");
    require(backend.q8_0_matmul_silu_mul(
                *reference_mid, *positive_device, *negative_device,
                *reference_norm, prefill_batch, cols, rows),
            "NVFP4 reference prefill SwiGLU");
    require(backend.q8_0_matmul(
                *reference_out, *down_device, *reference_mid,
                prefill_batch, rows, rows),
            "NVFP4 reference prefill down");

    auto fused_out = backend.tensor_f32(
        static_cast<uint64_t>(prefill_batch) * rows,
        "nvfp4_ffn_fused_out");
    require(backend.nvfp4_ffn_prefill(
                *fused_out, *positive_device, *negative_device, *down_device,
                *norm_weight, *prefill_x, prefill_batch, cols, rows, rows,
                norm_eps),
            "NVFP4 fused prefill FFN");
    std::vector<float> got_reference(
        static_cast<size_t>(prefill_batch) * rows);
    std::vector<float> got_fused(got_reference.size());
    require(backend.copy_to_host(
                *reference_out, got_reference.data(), 0,
                got_reference.size()),
            "copy NVFP4 reference FFN");
    require(backend.copy_to_host(
                *fused_out, got_fused.data(), 0, got_fused.size()),
            "copy NVFP4 fused FFN");
    require_close(got_fused, got_reference, 8.0f, 0.02f,
                  "NVFP4 fused prefill FFN");

    std::vector<__nv_bfloat16> prefill_input_bf16(prefill_input.size());
    std::vector<float> prefill_input_rounded(prefill_input.size());
    for (size_t i = 0; i < prefill_input.size(); ++i) {
        prefill_input_bf16[i] = __float2bfloat16(prefill_input[i]);
        prefill_input_rounded[i] =
            __bfloat162float(prefill_input_bf16[i]);
    }
    auto bf16_input = backend.tensor_bf16(
        prefill_input.size(), "nvfp4_ffn_bf16_input");
    auto bf16_output = backend.tensor_bf16(
        static_cast<uint64_t>(prefill_batch) * rows,
        "nvfp4_ffn_bf16_output");
    require(backend.copy_bytes_from_host(
                *bf16_input, 0, prefill_input_bf16.data(),
                prefill_input_bf16.size() * sizeof(__nv_bfloat16)),
            "copy NVFP4 BF16 FFN input");
    require(backend.nvfp4_ffn_prefill(
                *bf16_output, *positive_device, *negative_device,
                *down_device, *norm_weight, *bf16_input,
                prefill_batch, cols, rows, rows, norm_eps),
            "NVFP4 BF16 fused prefill FFN");

    auto rounded_input = backend.tensor_f32(
        prefill_input_rounded.size(), "nvfp4_ffn_rounded_input");
    auto rounded_norm = backend.tensor_f32(
        static_cast<uint64_t>(prefill_batch) * cols,
        "nvfp4_ffn_rounded_norm");
    auto rounded_mid = backend.tensor_f32(
        static_cast<uint64_t>(prefill_batch) * rows,
        "nvfp4_ffn_rounded_mid");
    auto rounded_out = backend.tensor_f32(
        static_cast<uint64_t>(prefill_batch) * rows,
        "nvfp4_ffn_rounded_out");
    require(backend.copy_bytes_from_host(
                *rounded_input, 0, prefill_input_rounded.data(),
                prefill_input_rounded.size() * sizeof(float)),
            "copy NVFP4 rounded FFN input");
    require(backend.rms_norm_batch(
                *rounded_norm, *rounded_input, *norm_weight,
                prefill_batch, cols, norm_eps),
            "NVFP4 rounded reference RMSNorm");
    require(backend.q8_0_matmul_silu_mul(
                *rounded_mid, *positive_device, *negative_device,
                *rounded_norm, prefill_batch, cols, rows),
            "NVFP4 rounded reference SwiGLU");
    require(backend.q8_0_matmul(
                *rounded_out, *down_device, *rounded_mid,
                prefill_batch, rows, rows),
            "NVFP4 rounded reference down");
    std::vector<float> got_rounded_reference(
        static_cast<size_t>(prefill_batch) * rows);
    require(backend.copy_to_host(
                *rounded_out, got_rounded_reference.data(), 0,
                got_rounded_reference.size()),
            "copy NVFP4 rounded reference FFN");
    const auto got_bf16_output =
        copy_bf16_to_float(backend, *bf16_output);
    require_close(got_bf16_output, got_rounded_reference,
                  16.0f, 0.03f,
                  "NVFP4 BF16 fused prefill FFN");
}

float decode_e2m1(uint8_t value) {
    static constexpr float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                            2.0f, 3.0f, 4.0f, 6.0f};
    const float result = magnitude[value & 0x7U];
    return value & 0x8U ? -result : result;
}

float decode_e4m3(uint8_t value) {
    __nv_fp8_e4m3 fp8;
    fp8.__x = value;
    return static_cast<float>(fp8);
}

void test_nvfp4_checkpoint(qw3::DeviceBackend &backend, const char *model_path) {
    auto model = qw3::open_model_source(model_path);
    const qw3::QwenTokenizer tokenizer(model_path);
    if (tokenizer.decode_one(tokenizer.eos_id()) != "<|im_end|>") {
        std::fprintf(stderr, "checkpoint tokenizer did not resolve chat EOS\n");
        std::exit(1);
    }
    const char *mtp_norm_names[] = {
        "blk.64.attn_norm.weight",
        "blk.64.post_attention_norm.weight",
        "blk.64.attn_q_norm.weight",
        "blk.64.attn_k_norm.weight",
        "blk.64.nextn.enorm.weight",
        "blk.64.nextn.hnorm.weight",
        "blk.64.nextn.shared_head_norm.weight",
    };
    for (const char *name : mtp_norm_names) {
        const qw3::ModelTensorInfo *norm = model->find_tensor(name);
        if (!norm || !norm->add_one) {
            std::fprintf(stderr, "checkpoint MTP norm transform missing: %s\n", name);
            std::exit(1);
        }
    }
    const qw3::ModelTensorInfo *tensor =
        model->find_tensor("blk.0.ffn_gate.weight");
    if (!tensor || tensor->type != qw3::ModelTensorType::NVFP4_E2M1 ||
        tensor->dims.size() != 2 || tensor->scale_dims.size() != 2) {
        std::fprintf(stderr, "checkpoint NVFP4 probe tensor is missing or malformed\n");
        std::exit(1);
    }
    const uint32_t cols = static_cast<uint32_t>(tensor->dims[0]);
    const uint32_t rows = static_cast<uint32_t>(tensor->dims[1]);
    const uint32_t scale_cols = static_cast<uint32_t>(tensor->scale_dims[0]);
    const uint32_t scale_rows = static_cast<uint32_t>(tensor->scale_dims[1]);
    auto weight = backend.weight_nvfp4_e2m1(
        tensor->data, tensor->scale_data, scale_rows, scale_cols,
        tensor->input_global_scale_divisor,
        tensor->weight_global_scale_divisor,
        rows, cols, tensor->name.c_str());

    std::vector<float> input(cols, 1.0f);
    const auto got = run_linear(backend, *weight, input, 1, cols, rows);

    const float input_divisor = tensor->input_global_scale_divisor;
    const __nv_fp8_e4m3 activation_scale_fp8(input_divisor / 6.0f);
    const float activation_scale = static_cast<float>(activation_scale_fp8);
    const float quant_multiplier = input_divisor / activation_scale;
    const uint8_t packed_input = __nv_cvt_float2_to_fp4x2(
        make_float2(quant_multiplier, quant_multiplier),
        __NV_E2M1, cudaRoundNearest);
    const float quantized_input = decode_e2m1(packed_input & 0xFU);

    const auto *packed = static_cast<const uint8_t *>(tensor->data);
    const auto *scales = static_cast<const uint8_t *>(tensor->scale_data);
    float max_absolute = 0.0f;
    float max_relative = 0.0f;
    for (uint32_t row = 0; row < std::min<uint32_t>(rows, 32); ++row) {
        float expected = 0.0f;
        for (uint32_t col = 0; col < cols; ++col) {
            const uint8_t byte = packed[static_cast<uint64_t>(row) * (cols / 2) + col / 2];
            const uint8_t nibble = col & 1U ? byte >> 4 : byte & 0xFU;
            const float weight_scale = decode_e4m3(
                scales[static_cast<uint64_t>(row) * scale_cols + col / 16]);
            expected += decode_e2m1(nibble) * weight_scale * quantized_input;
        }
        expected *= activation_scale /
                    (input_divisor * tensor->weight_global_scale_divisor);
        const float absolute = std::fabs(got[row] - expected);
        const float relative = absolute / std::max(1.0f, std::fabs(expected));
        max_absolute = std::max(max_absolute, absolute);
        max_relative = std::max(max_relative, relative);
        if (absolute > 0.25f && relative > 0.02f) {
            std::fprintf(stderr,
                         "checkpoint NVFP4 mismatch row %u: got=%g expected=%g abs=%g rel=%g\n",
                         row, got[row], expected, absolute, relative);
            std::exit(1);
        }
    }
    std::printf("NVFP4 checkpoint linear: max_abs=%g max_rel=%g\n",
                max_absolute, max_relative);

    constexpr uint32_t batch = 4;
    const float input_values[batch] = {1.0f, 0.5f, -0.75f, 2.0f};
    std::vector<float> batched_input(static_cast<size_t>(batch) * cols);
    for (uint32_t item = 0; item < batch; ++item) {
        std::fill(batched_input.begin() + static_cast<size_t>(item) * cols,
                  batched_input.begin() + static_cast<size_t>(item + 1) * cols,
                  input_values[item]);
    }
    const auto batched_got =
        run_linear(backend, *weight, batched_input, batch, cols, rows);
    max_absolute = 0.0f;
    max_relative = 0.0f;
    for (uint32_t item = 0; item < batch; ++item) {
        const float amax = std::fabs(input_values[item]);
        const __nv_fp8_e4m3 item_scale_fp8(
            input_divisor * (amax / 6.0f));
        const float item_scale = static_cast<float>(item_scale_fp8);
        const float item_multiplier =
            amax > 0.0f && item_scale > 0.0f
                ? input_divisor / item_scale
                : 0.0f;
        const uint8_t item_packed = __nv_cvt_float2_to_fp4x2(
            make_float2(input_values[item] * item_multiplier,
                        input_values[item] * item_multiplier),
            __NV_E2M1, cudaRoundNearest);
        const float item_quantized = decode_e2m1(item_packed & 0xfU);
        for (uint32_t row = 0; row < std::min<uint32_t>(rows, 32); ++row) {
            float expected = 0.0f;
            for (uint32_t col = 0; col < cols; ++col) {
                const uint8_t byte =
                    packed[static_cast<uint64_t>(row) * (cols / 2) + col / 2];
                const uint8_t nibble =
                    col & 1U ? byte >> 4 : byte & 0xfU;
                const float weight_scale = decode_e4m3(
                    scales[static_cast<uint64_t>(row) * scale_cols + col / 16]);
                expected +=
                    decode_e2m1(nibble) * weight_scale * item_quantized;
            }
            expected *= item_scale /
                        (input_divisor *
                         tensor->weight_global_scale_divisor);
            const float actual =
                batched_got[static_cast<size_t>(item) * rows + row];
            const float absolute = std::fabs(actual - expected);
            const float relative =
                absolute / std::max(1.0f, std::fabs(expected));
            max_absolute = std::max(max_absolute, absolute);
            max_relative = std::max(max_relative, relative);
            if (absolute > 0.25f && relative > 0.02f) {
                std::fprintf(
                    stderr,
                    "batched checkpoint NVFP4 mismatch item %u row %u: "
                    "got=%g expected=%g abs=%g rel=%g\n",
                    item, row, actual, expected, absolute, relative);
                std::exit(1);
            }
        }
    }
    std::printf(
        "NVFP4 checkpoint batched linear: max_abs=%g max_rel=%g\n",
        max_absolute, max_relative);
}

} // namespace

int main(int argc, char **argv) {
    auto backend = qw3::make_cuda_device_backend(qw3::LinearBackend::Auto);
    require(backend->begin(), "backend begin");
    test_bf16(*backend);
    test_fp8_shape(*backend, 4);
    test_fp8_shape(*backend, 128);
    test_fp8_shape(*backend, 512);
    test_fp8_bf16_prefill(*backend);
    test_fp8_fused_rms_fanout(*backend);
    test_fp8_fused_paths(*backend);
    test_nvfp4_shape(*backend, 1);
    test_nvfp4_shape(*backend, 2);
    test_nvfp4_shape(*backend, 4);
    test_nvfp4_shape(*backend, 8);
    test_nvfp4_shape(*backend, 128);
    test_nvfp4_fused_paths(*backend);
    if (argc > 1) test_nvfp4_checkpoint(*backend, argv[1]);
    require(backend->end(), "backend end");
    std::puts("mixed linear parity: PASS");
    return 0;
}
