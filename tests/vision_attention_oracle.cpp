#include "qw3/device_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void check(qw3::DeviceStatus status) {
    if (!status.ok) throw std::runtime_error(status.message);
}

uint16_t bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

double value(uint16_t bits) {
    uint32_t raw = static_cast<uint32_t>(bits) << 16;
    float result;
    std::memcpy(&result, &raw, sizeof(result));
    return result;
}

void compare(qw3::DeviceBackend &backend, uint32_t dim) {
    constexpr uint32_t padded = 128, heads = 16, tokens = 46;
    // Two images and an empty segment. Attention must remain noncausal within
    // an image and must never see tokens belonging to another image.
    const uint32_t boundaries[] = {0, 17, 17, tokens};
    const size_t count = static_cast<size_t>(tokens) * heads * padded;
    std::vector<uint16_t> q(count), k(count), v(count), actual(count);
    auto offset = [](uint32_t t, uint32_t h, uint32_t d) {
        return (static_cast<size_t>(t) * heads + h) * padded + d;
    };
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t h = 0; h < heads; ++h) {
            for (uint32_t d = 0; d < dim; ++d) {
                const size_t i = offset(t, h, d);
                q[i] = bf16(1.3f * std::sin(float(7*t + 3*h + d) * .17f));
                k[i] = bf16(1.1f * std::cos(float(3*t + h + 2*d) * .11f));
                v[i] = bf16(.4f * std::sin(float(t + h + 5*d) * .23f)
                            + (t < 17 ? -.2f : .2f));
            }
        }
    }
    auto qd = backend.tensor_bf16(count, "vision oracle Q");
    auto kd = backend.tensor_bf16(count, "vision oracle K");
    auto vd = backend.tensor_bf16(count, "vision oracle V");
    auto out = backend.tensor_bf16(count, "vision oracle output");
    check(backend.copy_bytes_from_host(*qd, 0, q.data(), count * 2));
    check(backend.copy_bytes_from_host(*kd, 0, k.data(), count * 2));
    check(backend.copy_bytes_from_host(*vd, 0, v.data(), count * 2));
    check(backend.vision_bf16_attention(*out, *qd, *kd, *vd, boundaries,
                                       3, heads, padded, 1.f / std::sqrt(float(dim))));
    check(backend.copy_bytes_to_host(*out, actual.data(), 0, count * 2));
    double max_abs = 0, square_sum = 0;
    for (uint32_t segment = 0; segment < 3; ++segment) {
        const uint32_t begin = boundaries[segment], end = boundaries[segment+1];
        for (uint32_t t = begin; t < end; ++t) {
            for (uint32_t h = 0; h < heads; ++h) {
                std::vector<double> scores(end - begin);
                for (uint32_t j = begin; j < end; ++j) {
                    double dot = 0;
                    for (uint32_t d = 0; d < dim; ++d)
                        dot += value(q[offset(t,h,d)]) * value(k[offset(j,h,d)]);
                    scores[j-begin] = dot / std::sqrt(double(dim));
                }
                const double maximum = *std::max_element(scores.begin(), scores.end());
                double sum = 0;
                for (double &s : scores) { s = std::exp(s - maximum); sum += s; }
                for (uint32_t d = 0; d < padded; ++d) {
                    double expected = 0;
                    for (uint32_t j = begin; j < end; ++j)
                        expected += scores[j-begin] * value(v[offset(j,h,d)]) / sum;
                    const double observed = value(actual[offset(t,h,d)]);
                    if (!std::isfinite(observed)) throw std::runtime_error("non-finite vision attention");
                    if (d >= dim && observed != 0) throw std::runtime_error("nonzero padded vision output");
                    const double error = std::abs(observed - expected);
                    max_abs = std::max(max_abs, error);
                    square_sum += error * error;
                }
            }
        }
    }
    const double rmse = std::sqrt(square_sum / (tokens * heads * dim));
    std::printf("vision_dim=%u padded=%u images=2 max_abs=%.9g rmse=%.9g\n",
                dim, padded, max_abs, rmse);
    // Values are bounded by 0.6. These absolute tolerances cover BF16 output
    // and attention-probability rounding, while comparing against an FP64
    // softmax oracle with the actual (not padded) model head dimension.
    if (max_abs > .003 || rmse > .001)
        throw std::runtime_error("vision attention differs from independent FP64 oracle");
}
} // namespace

int main() try {
    auto backend = qw3::make_cuda_device_backend(qw3::LinearBackend::Cublas);
    check(backend->begin());
    compare(*backend, 64);
    compare(*backend, 72);
    check(backend->end());
    return 0;
} catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
