#pragma once

#include "qw3/device_backend.hpp"
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <vector>

namespace qw3::detail {
// Keep verification on the decode-compatible Q8_1 quantizer and four-warp
// reduction, independent of how many requests the scheduler combines. Generic
// large GEMM uses different activation scales/reductions. This tiled reference
// is also useful when qualifying a faster batch-invariant implementation.
class Q8VerificationMath {
public:
    Q8VerificationMath(DeviceBackend &backend, bool enabled, bool allow_wide = true,
                       bool allow_wide_add = false, bool allow_mma = false)
        : backend_(backend), enabled_(enabled), allow_wide_(allow_wide),
          allow_wide_add_(allow_wide_add), allow_mma_(allow_mma) {}

    uint32_t wide_swiglu_calls() const { return wide_swiglu_calls_; }
    uint32_t wide_add_calls() const { return wide_add_calls_; }
    uint32_t mma_calls() const { return mma_calls_; }
    void set_mma_enabled(bool enabled) { allow_mma_ = enabled; }

    // Optional diagnostic, after all fanout outputs are ready. The observer
    // must leave outputs/input unchanged; normal execution has no observer.
    using ProjectionObserver = std::function<void(DeviceTensor *const *,
        const DeviceWeight *const *, const uint32_t *, uint32_t,
        const DeviceTensor &, uint32_t, uint32_t)>;
    void set_projection_observer(ProjectionObserver observer) { observer_ = std::move(observer); }

    DeviceStatus fanout(DeviceTensor *const *outs,
                        const DeviceWeight *const *weights,
                        const uint32_t *strides, uint32_t count,
                        const DeviceTensor &input, uint32_t rows, uint32_t input_stride) {
        if (!enabled_)
            return backend_.q8_0_matmul_fanout(outs, weights, strides, count, input, rows, input_stride);
        bool mma = allow_wide_ && allow_mma_ && rows > 0 && count > 0 && count <= 4 &&
            input.dtype == DeviceTensorDType::F32 && backend_.supports_q8_verify_mma();
        for (uint32_t i = 0; mma && i < count; ++i)
            mma = weights[i]->format == DeviceWeightFormat::Q8_0 &&
                weights[i]->cols % 256 == 0 && outs[i]->dtype == DeviceTensorDType::F32;
        if (mma) {
            auto status = backend_.q8_verify_fanout_mma(outs, weights, strides, count, input, rows, input_stride);
            if (status.ok) {
                mma_calls_ += count;
                if (observer_) observer_(outs, weights, strides, count, input, rows, input_stride);
            }
            return status;
        }
        bool wide = allow_wide_ && rows > 8 && count > 0 && count <= 4 &&
            input.dtype == DeviceTensorDType::F32 && backend_.supports_q8_verify_fanout();
        for (uint32_t i = 0; wide && i < count; ++i)
            wide = weights[i]->format == DeviceWeightFormat::Q8_0 &&
                outs[i]->dtype == DeviceTensorDType::F32;
        if (wide)
            return backend_.q8_verify_fanout(outs, weights, strides, count, input, rows, input_stride);
        for (uint32_t begin = 0; begin < rows; begin += 8) {
            const uint32_t n = std::min(8u, rows - begin);
            auto x = view(input, begin, n, input_stride);
            std::vector<std::unique_ptr<DeviceTensor>> owned;
            std::vector<DeviceTensor *> slices;
            for (uint32_t i = 0; i < count; ++i) {
                owned.push_back(view(*outs[i], begin, n, strides[i]));
                slices.push_back(owned.back().get());
            }
            auto status = backend_.q8_0_matmul_fanout(
                slices.data(), weights, strides, count, *x, n, input_stride);
            if (!status.ok) return status;
        }
        return {};
    }

    DeviceStatus matmul(DeviceTensor &out, const DeviceWeight &weight,
                        const DeviceTensor &input, uint32_t rows,
                        uint32_t input_stride, uint32_t output_stride) {
        if (!enabled_)
            return backend_.q8_0_matmul(out, weight, input, rows, input_stride, output_stride);
        DeviceTensor *outputs[] = {&out};
        const DeviceWeight *weights[] = {&weight};
        // The fanout API retains MMVQ at width eight; plain matmul switches
        // to generic GEMM at that width under the normal prefill threshold.
        return fanout(outputs, weights, &output_stride, 1, input, rows, input_stride);
    }

    DeviceStatus rms_fanout(DeviceTensor &normalized,
                           DeviceTensor *const *outs,
                           const DeviceWeight *const *weights,
                           const uint32_t *strides, uint32_t count,
                           const DeviceTensor &input, const DeviceWeight &norm,
                           uint32_t rows, uint32_t input_stride, float eps) {
        if (!enabled_)
            return backend_.rms_norm_q8_0_matmul_fanout(normalized, outs, weights, strides,
                count, input, norm, rows, input_stride, eps);
        auto status = backend_.rms_norm_batch(normalized, input, norm, rows, input_stride, eps);
        if (!status.ok) return status;
        return fanout(outs, weights, strides, count, normalized, rows, input_stride);
    }

    DeviceStatus silu_mul(DeviceTensor &out, const DeviceWeight &gate,
                          const DeviceWeight &up, const DeviceTensor &input,
                          uint32_t rows, uint32_t input_stride, uint32_t output_stride,
                          DeviceTensor *gate_scratch = nullptr, DeviceTensor *up_scratch = nullptr) {
        if (!enabled_)
            return backend_.q8_0_matmul_silu_mul(out, gate, up, input, rows, input_stride, output_stride);
        // Caller opts in by lending its existing FFN buffers. Reuse the wide
        // Q8_1 projection arithmetic, then run the same FP32 SwiGLU expression.
        // Packed outputs only: the elementwise kernel must not write padding.
        if (allow_wide_ && (rows > 8 || allow_mma_) && gate_scratch && up_scratch &&
            (backend_.supports_q8_verify_fanout() || (allow_mma_ && backend_.supports_q8_verify_mma())) &&
            gate.format == DeviceWeightFormat::Q8_0 && up.format == DeviceWeightFormat::Q8_0 &&
            gate.rows == up.rows && gate.cols == up.cols && output_stride == gate.rows &&
            input.dtype == DeviceTensorDType::F32 && out.dtype == DeviceTensorDType::F32 &&
            gate_scratch->dtype == DeviceTensorDType::F32 && up_scratch->dtype == DeviceTensorDType::F32) {
            DeviceTensor *outputs[] = {gate_scratch, up_scratch};
            const DeviceWeight *weights[] = {&gate, &up};
            const uint32_t strides[] = {output_stride, output_stride};
            auto status = fanout(outputs, weights, strides, 2, input, rows, input_stride);
            if (!status.ok) return status;
            status = backend_.silu_mul_n(out, *gate_scratch, *up_scratch,
                                         uint64_t(rows) * output_stride);
            if (status.ok) ++wide_swiglu_calls_;
            return status;
        }
        for (uint32_t begin = 0; begin < rows; begin += 8) {
            const uint32_t n = std::min(8u, rows - begin);
            auto x = view(input, begin, n, input_stride), y = view(out, begin, n, output_stride);
            auto status = backend_.q8_0_matmul_silu_mul(*y, gate, up, *x, n, input_stride, output_stride);
            if (!status.ok) return status;
        }
        return {};
    }

    DeviceStatus add(DeviceTensor &out, const DeviceTensor &residual, DeviceTensor &scratch,
                     const DeviceWeight &weight, const DeviceTensor &input,
                     uint32_t rows, uint32_t input_stride, uint32_t output_stride) {
        if (!enabled_)
            return backend_.q8_0_matmul_add(out, residual, scratch, weight, input, rows,
                                          input_stride, output_stride);
        // The recurrent output projection used to remain in eight-row tiles
        // even when the other Q8 projections shared a wider activation batch.
        // Preserve decode's quantizer/reduction and FP32 residual addition;
        // the caller's existing projection scratch avoids another allocation.
        // add_n is packed, so padded outputs retain the original tiled path.
        if (allow_wide_ && (allow_wide_add_ || allow_mma_) && (rows > 8 || allow_mma_) &&
            (backend_.supports_q8_verify_fanout() || (allow_mma_ && backend_.supports_q8_verify_mma())) &&
            weight.format == DeviceWeightFormat::Q8_0 && output_stride == weight.rows &&
            input.dtype == DeviceTensorDType::F32 && out.dtype == DeviceTensorDType::F32 &&
            residual.dtype == DeviceTensorDType::F32 && scratch.dtype == DeviceTensorDType::F32 &&
            &out == &residual && &scratch != &out) {
            DeviceTensor *outputs[] = {&scratch};
            const DeviceWeight *weights[] = {&weight};
            auto status = fanout(outputs, weights, &output_stride, 1, input, rows, input_stride);
            if (!status.ok) return status;
            status = backend_.add_n(out, residual, scratch, uint64_t(rows) * output_stride);
            if (status.ok) ++wide_add_calls_;
            return status;
        }
        for (uint32_t begin = 0; begin < rows; begin += 8) {
            const uint32_t n = std::min(8u, rows - begin);
            auto x = view(input, begin, n, input_stride), y = view(out, begin, n, output_stride);
            auto r = view(residual, begin, n, output_stride), temp = view(scratch, begin, n, output_stride);
            auto status = backend_.q8_0_matmul_add(*y, *r, *temp, weight, *x, n,
                                                 input_stride, output_stride);
            if (!status.ok) return status;
        }
        return {};
    }

private:
    std::unique_ptr<DeviceTensor> view(const DeviceTensor &tensor, uint32_t begin,
                                       uint32_t rows, uint32_t stride) {
        auto result = backend_.tensor_view(const_cast<DeviceTensor &>(tensor),
            uint64_t(begin) * stride * tensor.elem_size, uint64_t(rows) * stride,
            tensor.elem_size, tensor.dtype, "q8-verification-tile");
        if (!result) throw std::runtime_error("Q8 verification tile is out of bounds");
        return result;
    }
    DeviceBackend &backend_;
    bool enabled_;
    bool allow_wide_;
    bool allow_wide_add_;
    bool allow_mma_;
    uint32_t wide_swiglu_calls_ = 0;
    uint32_t wide_add_calls_ = 0;
    uint32_t mma_calls_ = 0;
    ProjectionObserver observer_;
};
} // namespace qw3::detail
