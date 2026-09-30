#pragma once

#include "q8_verification_math.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>

namespace qw3::detail {
// Explicit first-block diagnosis, never enabled during throughput measurement.
// Independent old-MMVQ projections see the candidate's exact current input;
// they neither replay recurrent state nor replace the candidate's outputs.
class Q8VerificationDiagnostics {
public:
    struct Span {
        uint64_t id, position, prompt_tokens;
        uint32_t begin, count;
        std::string tag;
        std::vector<uint32_t> tokens;
    };

    Q8VerificationDiagnostics(DeviceBackend &backend, const char *prefix,
                              const std::vector<Span> &spans, uint32_t hidden, uint32_t layers)
        : backend_(backend), reference_(backend, true, false), hidden_(hidden) {
        if (!prefix || !*prefix) return;
        for (const auto &span : spans) {
            if (span.tokens.size() != span.count) throw std::runtime_error("invalid verification trace span");
            const std::string path = std::string(prefix) + "." + std::to_string(span.id);
            if (std::ifstream(path + ".layers.bin").good() || std::ifstream(path + ".projections.csv").good())
                throw std::runtime_error("refusing to overwrite verification diagnostic: " + path);
            Trace trace{span, std::ofstream(path + ".layers.bin", std::ios::binary),
                        std::ofstream(path + ".projections.csv")};
            if (!trace.layers || !trace.projections) throw std::runtime_error("cannot create verification diagnostic");
            trace.layers.write("QW3LY001", 8);
            const uint64_t header[] = {span.id, span.position, span.prompt_tokens,
                                      span.count, hidden, layers, span.tag.size()};
            trace.layers.write(reinterpret_cast<const char *>(header), sizeof(header));
            trace.layers.write(span.tag.data(), span.tag.size());
            trace.layers.write(reinterpret_cast<const char *>(span.tokens.data()), span.tokens.size() * sizeof(uint32_t));
            trace.projections << "layer,projection,rows,cols,batch,max_abs,rms,relative_l2,max_reference,changed,tolerance_ratio\n"
                              << std::setprecision(12);
            traces_.push_back(std::move(trace));
        }
    }

    bool enabled() const { return !traces_.empty(); }
    void layer(uint32_t index) { layer_ = index; }

    // stage 0=embedding, 1=after recurrent/attention mixer, 2=after FFN.
    void hidden(const DeviceTensor &tensor, uint32_t stage) {
        for (auto &trace : traces_) {
            const auto values = read(tensor, uint64_t(trace.span.begin) * hidden_,
                                     uint64_t(trace.span.count) * hidden_);
            const uint64_t header[] = {layer_, stage};
            trace.layers.write(reinterpret_cast<const char *>(header), sizeof(header));
            trace.layers.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
            if (!trace.layers) throw std::runtime_error("verification layer trace write failed");
        }
    }

    void projections(DeviceTensor *const *outs, const DeviceWeight *const *weights,
                     const uint32_t *strides, uint32_t count, const DeviceTensor &input,
                     uint32_t batch, uint32_t input_stride) {
        for (uint32_t j = 0; j < count; ++j, ++projection_) {
            const uint64_t elements = uint64_t(batch) * strides[j];
            if (!scratch_ || scratch_->count < elements)
                scratch_ = backend_.scratch_f32(elements, "q8_verify_diagnostic_reference");
            require(reference_.matmul(*scratch_, *weights[j], input, batch, input_stride, strides[j]));
            for (auto &trace : traces_) {
                if (uint64_t(trace.span.begin) + trace.span.count > batch)
                    throw std::runtime_error("verification diagnostic projection row range");
                const auto offset = uint64_t(trace.span.begin) * strides[j];
                const auto size = uint64_t(trace.span.count) * strides[j];
                const auto ref = read(*scratch_, offset, size), actual = read(*outs[j], offset, size);
                double max_abs = 0, sum_error = 0, sum_ref = 0, max_ref = 0, tolerance_ratio = 0;
                uint64_t changed = 0, n = 0;
                for (uint32_t row = 0; row < trace.span.count; ++row)
                    for (uint64_t col = 0; col < weights[j]->rows; ++col) {
                        const auto index = uint64_t(row) * strides[j] + col;
                        const double a = actual[index], b = ref[index], error = std::abs(a - b);
                        if (!std::isfinite(a) || !std::isfinite(b))
                            throw std::runtime_error("nonfinite verification projection diagnostic");
                        max_abs = std::max(max_abs, error); max_ref = std::max(max_ref, std::abs(b));
                        sum_error += error * error; sum_ref += b * b; changed += a != b; ++n;
                        tolerance_ratio = std::max(tolerance_ratio, error / (1e-4 + 1e-5 * std::abs(b)));
                    }
                trace.projections << layer_ << ',' << projection_ << ',' << weights[j]->rows << ','
                    << weights[j]->cols << ',' << trace.span.count << ',' << max_abs << ','
                    << std::sqrt(sum_error / n) << ',' << std::sqrt(sum_error / std::max(1e-30, sum_ref))
                    << ',' << max_ref << ',' << changed << ',' << tolerance_ratio << '\n';
                if (!trace.projections) throw std::runtime_error("verification projection trace write failed");
            }
        }
    }

private:
    static void require(const DeviceStatus &status) {
        if (!status.ok) throw std::runtime_error(status.message);
    }
    std::vector<float> read(const DeviceTensor &tensor, uint64_t offset, uint64_t count) {
        std::vector<float> values(count);
        require(backend_.copy_to_host(tensor, values.data(), offset, count));
        return values;
    }
    struct Trace { Span span; std::ofstream layers, projections; };
    DeviceBackend &backend_;
    Q8VerificationMath reference_;
    uint32_t hidden_, layer_ = std::numeric_limits<uint32_t>::max(), projection_ = 0;
    std::vector<Trace> traces_;
    std::unique_ptr<DeviceTensor> scratch_;
};
} // namespace qw3::detail
