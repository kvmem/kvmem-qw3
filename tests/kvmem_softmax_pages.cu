// Exact softmax-over-mean-k parity test.
//
// Compares the original <=8192-page shared-memory kernel with the scalable
// tiled two-dot path, checks the 8192/8193 dispatch boundary, and validates the
// large path against a host reference including kept-band masking and subblocks.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace qw3 {
namespace ported {
bool launch_block_attn_score_softmax_pages(
    float *score, const float *q_multi, const float *kbar_multi,
    uint32_t n_layers, uint32_t n_tokens, uint32_t q_layer_stride,
    uint32_t n_blocks, uint32_t kbar_layer_stride, uint32_t n_heads,
    uint32_t n_kv_heads, uint32_t head_dim, float scale,
    uint32_t excl_lo_end, uint32_t excl_hi_begin, uint32_t n_subblocks,
    uint32_t reduce_max, cudaStream_t stream);
}
}

#define CHECK(call)                                                        \
    do {                                                                   \
        cudaError_t _err = (call);                                         \
        if (_err != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(_err));              \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

struct Case {
    uint32_t blocks;
    uint32_t subblocks;
    uint32_t excl_lo;
    uint32_t excl_hi;
    bool reduce_max;
};

static std::vector<float> host_reference(
        const std::vector<float> &q,
        const std::vector<float> &kbar,
        uint32_t layers,
        uint32_t tokens,
        uint32_t q_stride,
        uint32_t blocks,
        uint32_t kbar_stride,
        uint32_t heads,
        uint32_t kv_heads,
        uint32_t dim,
        float scale,
        uint32_t excl_lo,
        uint32_t excl_hi,
        uint32_t subblocks,
        bool reduce_max) {
    std::vector<double> accum(blocks, 0.0);
    const uint32_t group = heads / kv_heads;
    const uint32_t total = blocks * subblocks;
    std::vector<double> logits(total);
    for (uint32_t l = 0; l < layers; ++l) {
        for (uint32_t t = 0; t < tokens; ++t) {
            for (uint32_t h = 0; h < heads; ++h) {
                const uint32_t kh = h / group;
                double m = -INFINITY;
                for (uint32_t p = 0; p < total; ++p) {
                    const uint32_t w = p / subblocks;
                    if (w < excl_lo || w >= excl_hi) {
                        logits[p] = -INFINITY;
                        continue;
                    }
                    double dot = 0.0;
                    for (uint32_t d = 0; d < dim; ++d) {
                        const uint64_t qi =
                            ((static_cast<uint64_t>(l) * q_stride + t) * heads + h) *
                                dim + d;
                        const uint64_t ki =
                            (((static_cast<uint64_t>(l) * kbar_stride * subblocks + p) *
                                  kv_heads + kh) * dim) + d;
                        dot += static_cast<double>(q[qi]) * kbar[ki];
                    }
                    logits[p] = dot * scale;
                    m = std::max(m, logits[p]);
                }
                double sum = 0.0;
                for (double x : logits) {
                    if (std::isfinite(x)) sum += std::exp(x - m);
                }
                if (!(sum > 0.0)) continue;
                for (uint32_t w = excl_lo; w < std::min(excl_hi, blocks); ++w) {
                    double mass = 0.0;
                    for (uint32_t sb = 0; sb < subblocks; ++sb) {
                        const double v = std::exp(logits[w * subblocks + sb] - m) / sum;
                        mass = reduce_max && subblocks > 1 ? std::max(mass, v)
                                                          : mass + v;
                    }
                    accum[w] += mass;
                }
            }
        }
    }
    const double weight = 1.0 / (static_cast<double>(layers) * heads);
    std::vector<float> out(blocks);
    for (uint32_t w = 0; w < blocks; ++w)
        out[w] = static_cast<float>(accum[w] * weight);
    return out;
}

static std::vector<float> run_gpu(
        float *d_score,
        const float *d_q,
        const float *d_kbar,
        uint32_t layers,
        uint32_t tokens,
        uint32_t q_stride,
        uint32_t blocks,
        uint32_t kbar_stride,
        uint32_t heads,
        uint32_t kv_heads,
        uint32_t dim,
        float scale,
        const Case &tc,
        bool force_tiled) {
    if (force_tiled)
        setenv("QW3_KVMEM_SOFTMAX_PAGES_FORCE_TILED", "1", 1);
    else
        unsetenv("QW3_KVMEM_SOFTMAX_PAGES_FORCE_TILED");
    if (!qw3::ported::launch_block_attn_score_softmax_pages(
            d_score, d_q, d_kbar, layers, tokens, q_stride, blocks,
            kbar_stride, heads, kv_heads, dim, scale, tc.excl_lo, tc.excl_hi,
            tc.subblocks, tc.reduce_max ? 1u : 0u, 0)) {
        std::fprintf(stderr, "launcher rejected blocks=%u subblocks=%u\n",
                     blocks, tc.subblocks);
        std::exit(1);
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<float> out(blocks);
    CHECK(cudaMemcpy(out.data(), d_score, blocks * sizeof(float),
                     cudaMemcpyDeviceToHost));
    return out;
}

static void compare(const char *label,
                    const std::vector<float> &a,
                    const std::vector<float> &b,
                    double abs_limit,
                    double rel_limit) {
    double max_abs = 0.0;
    double max_rel = 0.0;
    uint32_t worst = 0;
    for (uint32_t i = 0; i < a.size(); ++i) {
        const double abs_err = std::fabs(static_cast<double>(a[i]) - b[i]);
        const double rel_err = abs_err / (std::fabs(static_cast<double>(b[i])) + 1e-12);
        if (abs_err > max_abs) worst = i;
        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, rel_err);
    }
    std::printf("[kvmem-softmax-pages] %-30s max_abs=%.3e max_rel=%.3e worst=%u\n",
                label, max_abs, max_rel, worst);
    if (max_abs > abs_limit && max_rel > rel_limit) {
        std::fprintf(stderr, "FAIL %s: a[%u]=%.9g b[%u]=%.9g\n", label,
                     worst, a[worst], worst, b[worst]);
        std::exit(1);
    }
}

static void run_case(const Case &tc) {
    constexpr uint32_t layers = 2;
    constexpr uint32_t tokens = 3;
    constexpr uint32_t q_stride = 5;
    constexpr uint32_t heads = 4;
    constexpr uint32_t kv_heads = 2;
    constexpr uint32_t dim = 8;
    const uint32_t kbar_stride = tc.blocks + 7;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));

    const uint64_t q_count =
        static_cast<uint64_t>(layers) * q_stride * heads * dim;
    const uint64_t k_count =
        static_cast<uint64_t>(layers) * kbar_stride * tc.subblocks * kv_heads * dim;
    std::mt19937 rng(1000 + tc.blocks + tc.subblocks * 17);
    std::normal_distribution<float> nd(0.0f, 0.35f);
    std::vector<float> q(q_count), kbar(k_count);
    for (float &v : q) v = nd(rng);
    for (float &v : kbar) v = nd(rng);

    float *d_q = nullptr, *d_kbar = nullptr, *d_score = nullptr;
    CHECK(cudaMalloc(&d_q, q_count * sizeof(float)));
    CHECK(cudaMalloc(&d_kbar, k_count * sizeof(float)));
    CHECK(cudaMalloc(&d_score, tc.blocks * sizeof(float)));
    CHECK(cudaMemcpy(d_q, q.data(), q_count * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_kbar, kbar.data(), k_count * sizeof(float),
                     cudaMemcpyHostToDevice));

    const std::vector<float> ref = host_reference(
        q, kbar, layers, tokens, q_stride, tc.blocks, kbar_stride, heads,
        kv_heads, dim, scale, tc.excl_lo, tc.excl_hi, tc.subblocks,
        tc.reduce_max);
    const std::vector<float> tiled = run_gpu(
        d_score, d_q, d_kbar, layers, tokens, q_stride, tc.blocks,
        kbar_stride, heads, kv_heads, dim, scale, tc, true);
    char label[128];
    std::snprintf(label, sizeof(label), "tiled-vs-host B=%u ns=%u max=%u",
                  tc.blocks, tc.subblocks, tc.reduce_max ? 1u : 0u);
    compare(label, tiled, ref, 3e-6, 3e-3);

    if (static_cast<uint64_t>(tc.blocks) * tc.subblocks <= 8192) {
        const std::vector<float> fused = run_gpu(
            d_score, d_q, d_kbar, layers, tokens, q_stride, tc.blocks,
            kbar_stride, heads, kv_heads, dim, scale, tc, false);
        std::snprintf(label, sizeof(label), "tiled-vs-fused B=%u ns=%u max=%u",
                      tc.blocks, tc.subblocks, tc.reduce_max ? 1u : 0u);
        compare(label, tiled, fused, 3e-6, 3e-3);
    }

    CHECK(cudaFree(d_q));
    CHECK(cudaFree(d_kbar));
    CHECK(cudaFree(d_score));
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::fprintf(stderr, "[kvmem-softmax-pages] no CUDA device; skipping\n");
        return 0;
    }
    run_case({512, 1, 0, UINT32_MAX, false});
    run_case({4096, 1, 0, UINT32_MAX, false});
    run_case({8192, 1, 0, UINT32_MAX, false});
    run_case({8193, 1, 0, UINT32_MAX, false});
    run_case({4100, 2, 3, 4095, false});
    run_case({4100, 2, 3, 4095, true});
    unsetenv("QW3_KVMEM_SOFTMAX_PAGES_FORCE_TILED");
    std::printf("[kvmem-softmax-pages] PASS\n");
    return 0;
}
