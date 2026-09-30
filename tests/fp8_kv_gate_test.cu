#include "qw3/device_backend.hpp"
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void check(qw3::DeviceStatus s) { if (!s.ok) throw std::runtime_error(s.message); }
void run(qw3::DeviceBackend &d, unsigned batch, unsigned base, bool gated) {
    constexpr unsigned dim = 256, heads = 24, kv_heads = 4;
    constexpr unsigned width = kv_heads * dim, qstride = heads * 2 * dim;
    constexpr unsigned ostride = heads * dim + 11;
    constexpr float guard = -12345.f;
    const unsigned length = base + batch;
    std::vector<__nv_fp8_e4m3> values(length * width);
    for (unsigned t = 0; t < length; ++t)
        for (unsigned h = 0; h < kv_heads; ++h)
            std::fill_n(values.begin() + t * width + h * dim, dim,
                        __nv_fp8_e4m3(1.25f + h * .25f));
    std::vector<float> query(batch * qstride), result(batch * ostride, guard);
    for (unsigned b = 0; b < batch; ++b)
        for (unsigned h = 0; h < heads; ++h)
            for (unsigned c = 0; c < dim; ++c)
                query[b * qstride + h * 2 * dim + dim + c] =
                    .2f * std::sin(float(b + h * 3 + c));
    auto k = d.tensor_fp8_kv(values.size(), "zero K");
    auto v = d.tensor_fp8_kv(values.size(), "constant V by head");
    auto q = d.tensor_f32(query.size(), "Q+gate");
    auto o = d.tensor_f32(result.size(), "guarded O");
    check(d.copy_bytes_from_host(*v, 0, values.data(), values.size()));
    check(d.copy_bytes_from_host(*q, 0, query.data(), query.size() * 4));
    check(d.copy_bytes_from_host(*o, 0, result.data(), result.size() * 4));
    check(gated
        ? d.attention_decode_batch_gated(*o, *q, 2 * dim, *k, *v, heads,
            kv_heads, dim, base, batch, qstride, ostride, 1.f / 16)
        : d.attention_decode_batch(*o, *q, 2 * dim, *k, *v, heads,
            kv_heads, dim, base, batch, qstride, ostride, 1.f / 16));
    check(d.copy_to_host(*o, result.data(), 0, result.size()));
    float max_error = 0;
    for (unsigned b = 0; b < batch; ++b) {
        for (unsigned c = 0; c < ostride; ++c) {
            const float actual = result[b * ostride + c];
            if (c >= heads * dim) {
                if (actual != guard) throw std::runtime_error("FP8 output row guard overwritten");
                continue;
            }
            const unsigned h = c / dim, col = c % dim;
            float expected = 1.25f + (h / (heads / kv_heads)) * .25f;
            if (gated) expected /= 1.f + std::exp(-query[b * qstride + h * 2 * dim + dim + col]);
            if (!std::isfinite(actual)) throw std::runtime_error("non-finite FP8 attention");
            max_error = std::max(max_error, std::abs(actual - expected));
        }
    }
    std::printf("FP8 causal batch=%u base=%u gated=%d max_abs=%g\n", batch, base, int(gated), max_error);
    if (max_error > .002f)
        throw std::runtime_error("FP8 attention gate applied at the wrong boundary");
}
}
int main() try {
    auto d = qw3::make_cuda_device_backend(qw3::LinearBackend::Cublas);
    check(d->begin());
    for (unsigned base : {65u, 8193u})
        for (unsigned batch : {1u, 2u, 7u, 8u, 9u, 33u})
            for (bool gated : {false, true}) run(*d, batch, base, gated);
    check(d->end());
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
