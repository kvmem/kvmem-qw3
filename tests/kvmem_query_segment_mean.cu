// Query segment mean-Q parity test.
//
// Verifies that streaming chunk updates produce the same de-RoPE'd segment
// means as a CPU FP32 reference, including segments split across chunks and the
// near-threshold case where almost every segment contains one token.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace qw3::ported {
bool launch_derope_query_segment_mean_f16(
    void *q_mean, const float *q, uint32_t q_token_stride,
    uint32_t q_head_stride, uint32_t cnt, uint32_t query_offset,
    uint32_t query_tokens, uint32_t prototypes, uint32_t n_heads,
    uint32_t head_dim, uint32_t rope_dim, int32_t start_pos, float theta,
    cudaStream_t stream);
}

#define CHECK(call) do {                                                   \
    cudaError_t err_ = (call);                                             \
    if (err_ != cudaSuccess) {                                             \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(err_));                            \
        std::exit(1);                                                      \
    }                                                                      \
} while (0)

static void rotate(float *row, uint32_t rope_dim, int32_t position,
                   float theta) {
    const uint32_t half = rope_dim / 2;
    std::vector<float> copy(row, row + rope_dim);
    for (uint32_t i = 0; i < half; ++i) {
        const float inv_freq = std::pow(
            theta, -2.0f * static_cast<float>(i) /
                       static_cast<float>(rope_dim));
        const float angle = static_cast<float>(position) * inv_freq;
        const float s = std::sin(angle);
        const float c = std::cos(angle);
        row[i] = copy[i] * c - copy[i + half] * s;
        row[i + half] = copy[i] * s + copy[i + half] * c;
    }
}

static void run_case(uint32_t tokens, uint32_t prototypes,
                     const std::vector<uint32_t> &chunk_ends,
                     const char *label) {
    constexpr uint32_t heads = 2;
    constexpr uint32_t dim = 8;
    constexpr uint32_t rope_dim = 8;
    constexpr uint32_t q_head_stride = 2 * dim;
    constexpr uint32_t q_token_stride = heads * q_head_stride;
    constexpr int32_t position_base = 37;
    constexpr float theta = 10000.0f;

    std::mt19937 rng(1701 + tokens * 7 + prototypes);
    std::normal_distribution<float> nd(0.0f, 0.4f);
    std::vector<float> content(
        static_cast<size_t>(tokens) * heads * dim);
    std::vector<float> rotated(
        static_cast<size_t>(tokens) * q_token_stride, 0.0f);
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t h = 0; h < heads; ++h) {
            float *content_row = content.data() +
                (static_cast<size_t>(t) * heads + h) * dim;
            float *rotated_row = rotated.data() +
                static_cast<size_t>(t) * q_token_stride +
                static_cast<size_t>(h) * q_head_stride;
            for (uint32_t d = 0; d < dim; ++d) {
                content_row[d] = nd(rng);
                rotated_row[d] = content_row[d];
            }
            rotate(rotated_row, rope_dim,
                   position_base + static_cast<int32_t>(t), theta);
        }
    }

    std::vector<float> expected(
        static_cast<size_t>(prototypes) * heads * dim, 0.0f);
    for (uint32_t p = 0; p < prototypes; ++p) {
        const uint32_t begin = static_cast<uint32_t>(
            (static_cast<uint64_t>(p) * tokens) / prototypes);
        const uint32_t end = static_cast<uint32_t>(
            (static_cast<uint64_t>(p + 1) * tokens) / prototypes);
        for (uint32_t t = begin; t < end; ++t) {
            for (uint32_t h = 0; h < heads; ++h) {
                for (uint32_t d = 0; d < dim; ++d) {
                    expected[(static_cast<size_t>(p) * heads + h) * dim + d] +=
                        content[(static_cast<size_t>(t) * heads + h) * dim + d];
                }
            }
        }
        const float inv = 1.0f / static_cast<float>(end - begin);
        for (uint32_t h = 0; h < heads; ++h) {
            for (uint32_t d = 0; d < dim; ++d) {
                expected[(static_cast<size_t>(p) * heads + h) * dim + d] *= inv;
            }
        }
    }

    float *d_q = nullptr;
    __half *d_mean = nullptr;
    CHECK(cudaMalloc(&d_q, rotated.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_mean, expected.size() * sizeof(__half)));
    CHECK(cudaMemcpy(d_q, rotated.data(), rotated.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_mean, 0, expected.size() * sizeof(__half)));

    uint32_t begin = 0;
    for (uint32_t end : chunk_ends) {
        if (end <= begin || end > tokens) {
            std::fprintf(stderr, "invalid chunk boundary for %s\n", label);
            std::exit(1);
        }
        const uint32_t count = end - begin;
        const bool ok = qw3::ported::launch_derope_query_segment_mean_f16(
            d_mean, d_q + static_cast<size_t>(begin) * q_token_stride,
            q_token_stride, q_head_stride, count, begin, tokens, prototypes,
            heads, dim, rope_dim, position_base + static_cast<int32_t>(begin),
            theta, 0);
        if (!ok) {
            std::fprintf(stderr, "launcher rejected %s chunk [%u,%u)\n",
                         label, begin, end);
            std::exit(1);
        }
        CHECK(cudaDeviceSynchronize());
        begin = end;
    }
    if (begin != tokens) {
        std::fprintf(stderr, "chunks do not cover %s\n", label);
        std::exit(1);
    }

    std::vector<__half> got_half(expected.size());
    CHECK(cudaMemcpy(got_half.data(), d_mean,
                     got_half.size() * sizeof(__half),
                     cudaMemcpyDeviceToHost));
    double max_abs = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double error = std::fabs(
            static_cast<double>(__half2float(got_half[i])) - expected[i]);
        if (error > max_abs) {
            max_abs = error;
            worst = i;
        }
    }
    std::printf("[kvmem-query-segment-mean] %-20s max_abs=%.3e\n",
                label, max_abs);
    if (max_abs > 1.5e-3) {
        std::fprintf(stderr,
                     "FAIL %s at %zu: got %.9g want %.9g\n",
                     label, worst, __half2float(got_half[worst]),
                     expected[worst]);
        std::exit(1);
    }

    CHECK(cudaFree(d_q));
    CHECK(cudaFree(d_mean));
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::fprintf(stderr,
                     "[kvmem-query-segment-mean] no CUDA device; skipping\n");
        return 0;
    }
    run_case(19, 5, {3, 11, 19}, "cross-chunk");
    run_case(513, 512, {127, 256, 384, 513}, "threshold-plus-one");
    run_case(1025, 512, {128, 511, 777, 1025}, "uneven-segments");
    return 0;
}
