// CUDA native attention regression: run with memcheck AND racecheck.
// Use separate processes for QW3_PREFILL_ATTN / QW3_PREFILL_FA2_* variants;
// dispatch settings are cached on first use. No model weights are required.
#include "qw3/device_backend.hpp"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace qw3::ported {
size_t fattn_prefill_mma_gqa_v2_scratch_bytes(uint32_t batch, uint32_t n_heads,
                                            uint32_t n_kv_heads, uint32_t head_dim);
}

namespace {
void check(qw3::DeviceStatus s) {
    if (!s.ok) throw std::runtime_error(s.message);
}

void compare(qw3::DeviceBackend &backend, unsigned dim, unsigned batch,
             unsigned base, bool fp32, float q_scale = 1.0f) {
    constexpr unsigned heads = 24, kv_heads = 4, guard = 8;
    constexpr float sentinel = 12345.0f;
    // v2 only instantiates one or two splits, even for unsupported env overrides.
    // A mismatched grid can launch duplicate writers without a numerical failure.
    const size_t scratch = qw3::ported::fattn_prefill_mma_gqa_v2_scratch_bytes(
        batch, heads, kv_heads, dim);
    const size_t two_splits = size_t(heads) * batch * 2 * (dim + 2) * sizeof(float);
    if (scratch != 0 && scratch != two_splits)
        throw std::runtime_error("v2 scratch split count has no matching kernel");
    const unsigned stride = kv_heads * dim, q_stride = heads * 2 * dim;
    const unsigned length = base + batch, out_stride = heads * dim + guard;
    std::vector<float> k(length * stride), v(k.size()), q(batch * q_stride);
    std::vector<__half> kh(k.size()), vh(k.size());
    uint32_t random = 0x13a598b7;
    auto sample = [&]() {
        random ^= random << 13; random ^= random >> 17; random ^= random << 5;
        return float(random & 65535) / 65536.0f - 0.5f;
    };
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = sample(); v[i] = sample();
        kh[i] = __float2half(k[i]); vh[i] = __float2half(v[i]);
        if (!fp32) { k[i] = __half2float(kh[i]); v[i] = __half2float(vh[i]); }
    }
    for (float &value : q) value = sample() * q_scale;
    std::vector<float> expected(batch * out_stride, sentinel);
    for (unsigned b = 0; b < batch; ++b) {
        for (unsigned h = 0; h < heads; ++h) {
            const unsigned kv_head = h / (heads / kv_heads);
            const float *query = q.data() + b * q_stride + h * 2 * dim;
            std::vector<double> scores(base + b + 1);
            for (unsigned t = 0; t < scores.size(); ++t) {
                const float *key = k.data() + t * stride + kv_head * dim;
                double dot = 0;
                for (unsigned d = 0; d < dim; ++d) dot += double(query[d]) * key[d];
                scores[t] = dot / std::sqrt(double(dim));
            }
            const double maximum = *std::max_element(scores.begin(), scores.end());
            double sum = 0;
            for (double &s : scores) { s = std::exp(s - maximum); sum += s; }
            for (unsigned d = 0; d < dim; ++d) {
                double value = 0;
                for (unsigned t = 0; t < scores.size(); ++t)
                    value += scores[t] * v[t * stride + kv_head * dim + d];
                expected[b * out_stride + h * dim + d] = float(value / sum);
            }
        }
    }
    auto kd = fp32 ? backend.tensor_f32(k.size(), "K") : backend.tensor_f16(k.size(), "K");
    auto vd = fp32 ? backend.tensor_f32(v.size(), "V") : backend.tensor_f16(v.size(), "V");
    auto qd = backend.tensor_f32(q.size(), "Q with gate stride");
    auto od = backend.tensor_f32(expected.size(), "O with row guards");
    check(backend.copy_bytes_from_host(*kd, 0, fp32 ? static_cast<void *>(k.data()) : kh.data(),
        k.size() * (fp32 ? sizeof(float) : sizeof(__half))));
    check(backend.copy_bytes_from_host(*vd, 0, fp32 ? static_cast<void *>(v.data()) : vh.data(),
        v.size() * (fp32 ? sizeof(float) : sizeof(__half))));
    check(backend.copy_bytes_from_host(*qd, 0, q.data(), q.size() * sizeof(float)));
    float max_error = 0;
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        std::vector<float> actual(expected.size(), sentinel);
        check(backend.copy_bytes_from_host(*od, 0, actual.data(), actual.size() * sizeof(float)));
        check(backend.attention_decode_batch(*od, *qd, 2 * dim, *kd, *vd,
            heads, kv_heads, dim, base, batch, q_stride, out_stride,
            1.0f / std::sqrt(float(dim))));
        check(backend.copy_to_host(*od, actual.data(), 0, actual.size()));
        for (unsigned b = 0; b < batch; ++b) {
            for (unsigned d = 0; d < out_stride; ++d) {
                const size_t i = b * out_stride + d;
                if (!std::isfinite(actual[i])) throw std::runtime_error("non-finite attention");
                if (d >= heads * dim && actual[i] != sentinel)
                    throw std::runtime_error("attention overwrote an output row guard");
                max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
            }
        }
    }
    std::printf("dtype=%s dim=%u batch=%u base=%u GQA=6 q_scale=%g repeats=3 max_abs_error=%g\n",
                fp32 ? "fp32" : "fp16", dim, batch, base, q_scale, max_error);
    if (max_error > (fp32 ? 1e-5f : 1e-3f))
        throw std::runtime_error("attention differs from double-precision CPU oracle");
}
} // namespace

int main(int argc, char **argv) try {
    bool fp32 = false;
    if (argc == 2 && std::strcmp(argv[1], "--fp32") == 0) fp32 = true;
    else if (argc != 1) throw std::runtime_error("usage: native_attention_oracle [--fp32]");
    auto backend = qw3::make_cuda_device_backend(qw3::LinearBackend::Cublas);
    check(backend->begin());
    compare(*backend, 256, 31, 0, fp32);   // partial query tile + empty KV split
    compare(*backend, 256, 17, 63, fp32);  // former shared-buffer race
    compare(*backend, 128, 17, 31, fp32);  // exercise MMA-128, not the tiny vec fallback
    compare(*backend, 256, 33, 997, fp32); // non-aligned prefix and several KV tiles
    compare(*backend, 128, 65, 127, fp32);
    compare(*backend, 256, 1, 0, fp32);    // masked rows and an empty split (min_batch=1)
    compare(*backend, 256, 17, 257, fp32, 16.0f); // sharper softmax across several KV tiles
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
