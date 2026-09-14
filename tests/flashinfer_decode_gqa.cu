#include "qw3/device_backend.hpp"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

void check(qw3::DeviceStatus status) {
    if (!status.ok) throw std::runtime_error(status.message);
}

template<class KV>
void compare(qw3::DeviceBackend &backend, unsigned heads, unsigned kv_heads, bool fp8) {
    constexpr unsigned dim = 256, batch = 4, base = 1001, page = 32;
    constexpr unsigned pages = (base + batch + page - 1) / page;
    const unsigned stride = kv_heads * dim, q_stride = heads * 2 * dim;
    std::vector<KV> k(pages * page * stride), v(k.size());
    std::vector<float> query(batch * q_stride), result(batch * heads * dim);
    std::vector<int32_t> mapping(pages);
    for (unsigned p = 0; p < pages; ++p) mapping[p] = pages - p - 1;
    for (unsigned t = 0; t < pages * page; ++t) {
        const size_t offset = (mapping[t / page] * page + t % page) * stride;
        for (unsigned c = 0; c < stride; ++c) {
            k[offset + c] = KV(0.5f * std::sin(float(t * 3 + c) * 0.13f));
            v[offset + c] = KV(0.5f * std::cos(float(t + c * 2) * 0.17f));
        }
    }
    for (unsigned i = 0; i < query.size(); ++i) query[i] = std::sin(float(i) * 0.11f);
    auto kd = fp8 ? backend.tensor_fp8_kv(k.size(), "K") : backend.tensor_f16(k.size(), "K");
    auto vd = fp8 ? backend.tensor_fp8_kv(v.size(), "V") : backend.tensor_f16(v.size(), "V");
    auto qd = backend.tensor_f32(query.size(), "Q+gate");
    auto out = backend.tensor_f32(result.size(), "attention");
    auto pd = backend.tensor_i32(pages, "permuted pages");
    check(backend.copy_bytes_from_host(*kd, 0, k.data(), k.size() * sizeof(KV)));
    check(backend.copy_bytes_from_host(*vd, 0, v.data(), v.size() * sizeof(KV)));
    check(backend.copy_bytes_from_host(*qd, 0, query.data(), query.size() * sizeof(float)));
    check(backend.copy_i32_from_host(*pd, 0, mapping.data(), pages));
    check(backend.attention_decode_batch_paged_gated_device(
        *out, *qd, 2 * dim, *kd, *vd, *pd, pages, page, heads, kv_heads, dim,
        base, batch, q_stride, heads * dim, 1.0f / std::sqrt(float(dim))));
    check(backend.copy_to_host(*out, result.data(), 0, result.size()));
    float max_error = 0;
    for (unsigned b = 0; b < batch; ++b) {
        const unsigned length = base + b + 1;
        for (unsigned h = 0; h < heads; ++h) {
            const unsigned kh = h / (heads / kv_heads);
            const size_t qi = b * q_stride + h * 2 * dim;
            std::vector<double> scores(length);
            for (unsigned t = 0; t < length; ++t) {
                const size_t offset = (mapping[t / page] * page + t % page) * stride + kh * dim;
                double dot = 0;
                for (unsigned c = 0; c < dim; ++c) dot += query[qi + c] * float(k[offset + c]);
                scores[t] = dot / std::sqrt(double(dim));
            }
            const double maximum = *std::max_element(scores.begin(), scores.end());
            double sum = 0;
            for (auto &score : scores) { score = std::exp(score - maximum); sum += score; }
            for (unsigned c = 0; c < dim; ++c) {
                double value = 0;
                for (unsigned t = 0; t < length; ++t) {
                    const size_t offset = (mapping[t / page] * page + t % page) * stride + kh * dim;
                    value += scores[t] * float(v[offset + c]);
                }
                value = value / sum / (1.0 + std::exp(-query[qi + dim + c]));
                const float actual = result[(b * heads + h) * dim + c];
                if (!std::isfinite(actual)) throw std::runtime_error("non-finite attention output");
                max_error = std::max(max_error, float(std::abs(actual - value)));
            }
        }
    }
    std::printf("GQA=%u heads=%u kv=%u dtype=%s max_abs_error=%g\n",
                heads / kv_heads, heads, kv_heads, fp8 ? "FP8" : "FP16", max_error);
    if (max_error > 0.001f) throw std::runtime_error("paged causal attention differs from CPU oracle");
}

int main() try {
    auto backend = qw3::make_cuda_device_backend(qw3::LinearBackend::Cublas);
    check(backend->begin());
    for (auto shape : {std::pair<unsigned, unsigned>{8, 2}, {16, 4}, {24, 4}}) {
        compare<__nv_fp8_e4m3>(*backend, shape.first, shape.second, true);
        compare<half>(*backend, shape.first, shape.second, false);
    }
    return 0;
} catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
