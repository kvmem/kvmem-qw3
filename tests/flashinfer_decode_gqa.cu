#include "qw3/device_backend.hpp"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

void check(qw3::DeviceStatus status) {
    if (!status.ok) throw std::runtime_error(status.message);
}

template<class KV>
void compare(qw3::DeviceBackend &backend, unsigned heads, unsigned kv_heads, bool fp8,
             bool prefill = false, unsigned dim = 256, unsigned batch = 4,
             unsigned base = 1001) {
    constexpr unsigned page = 32;
    const unsigned pages = (base + batch + page - 1) / page;
    const unsigned stride = kv_heads * dim, q_stride = heads * 2 * dim;
    const unsigned out_stride = heads * dim + 8;
    constexpr float sentinel = 12345.0f;
    std::vector<KV> k(pages * page * stride), v(k.size());
    std::vector<float> query(batch * q_stride), result(batch * out_stride, sentinel);
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
    std::vector<float> expected(result.size(), sentinel);
    for (unsigned b = 0; b < batch; ++b) {
        const unsigned length = base + b + 1;
        for (unsigned h = 0; h < heads; ++h) {
            const unsigned kh = h / (heads / kv_heads);
            const size_t qi = b * q_stride + h * 2 * dim;
            std::vector<double> scores(length);
            for (unsigned t = 0; t < length; ++t) {
                const size_t offset = (mapping[t / page] * page + t % page) * stride + kh * dim;
                double dot = 0;
                for (unsigned c = 0; c < dim; ++c) dot += double(query[qi + c]) * float(k[offset + c]);
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
                expected[b * out_stride + h * dim + c] = float(value);
            }
        }
    }
    float max_error = 0;
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        std::fill(result.begin(), result.end(), sentinel);
        check(backend.copy_bytes_from_host(*out, 0, result.data(), result.size() * sizeof(float)));
        const auto status = prefill
            ? backend.attention_prefill_batch_paged_gated_device(
                *out, *qd, 2 * dim, *kd, *vd, *pd, pages, page, heads, kv_heads, dim,
                base, batch, q_stride, out_stride, 1.0f / std::sqrt(float(dim)))
            : backend.attention_decode_batch_paged_gated_device(
                *out, *qd, 2 * dim, *kd, *vd, *pd, pages, page, heads, kv_heads, dim,
                base, batch, q_stride, out_stride, 1.0f / std::sqrt(float(dim)));
        check(status);
        check(backend.copy_to_host(*out, result.data(), 0, result.size()));
        for (size_t i = 0; i < result.size(); ++i) {
            if (!std::isfinite(result[i])) throw std::runtime_error("non-finite attention output");
            if (i % out_stride >= heads * dim && result[i] != sentinel)
                throw std::runtime_error("attention overwrote output row guard");
            max_error = std::max(max_error, std::abs(result[i] - expected[i]));
        }
    }
    std::printf("mode=%s dim=%u batch=%u base=%u GQA=%u heads=%u kv=%u dtype=%s repeats=3 max_abs_error=%g\n",
                prefill ? "prefill" : "decode", dim, batch, base,
                heads / kv_heads, heads, kv_heads, fp8 ? "FP8" : "FP16", max_error);
    if (max_error > 0.001f) throw std::runtime_error("paged causal attention differs from CPU oracle");
}

int main(int argc, char **argv) try {
    const bool prefill = argc == 2 && std::strcmp(argv[1], "--prefill") == 0;
    if (argc != 1 && !prefill) throw std::runtime_error("usage: flashinfer_decode_gqa [--prefill]");
    auto backend = qw3::make_cuda_device_backend(qw3::LinearBackend::Cublas);
    check(backend->begin());
    if (prefill) {
        for (unsigned dim : {128u, 256u}) {
            for (auto shape : {std::pair<unsigned, unsigned>{31, 0}, {33, 997}, {129, 63}}) {
                compare<__nv_fp8_e4m3>(*backend, 24, 4, true, true, dim, shape.first, shape.second);
                compare<half>(*backend, 24, 4, false, true, dim, shape.first, shape.second);
            }
        }
        return 0;
    }
    for (auto shape : {std::pair<unsigned, unsigned>{8, 2}, {16, 4}, {24, 4}}) {
        compare<__nv_fp8_e4m3>(*backend, shape.first, shape.second, true);
        compare<half>(*backend, shape.first, shape.second, false);
    }
    return 0;
} catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
