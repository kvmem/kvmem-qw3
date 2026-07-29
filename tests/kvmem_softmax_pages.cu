// Exact softmax-over-mean-k parity test.
//
// Compares the original <=8192-page shared-memory kernel with both scalable
// paths (bounded one-dot production path and retained two-dot A/B baseline),
// checks the 8192/8193 dispatch boundary, and validates against a host reference
// including kept-band masking and subblocks.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace qw3 {
namespace ported {
enum class KbarDType : uint32_t {
    F32 = 0,
    F16 = 1,
    FP8 = 2,
};
bool launch_block_attn_score_softmax_pages(
    float *score, const float *q_multi, const float *kbar_multi,
    uint32_t n_layers, uint32_t n_tokens, uint32_t q_layer_stride,
    uint32_t n_blocks, uint32_t kbar_layer_stride, uint32_t n_heads,
    uint32_t n_kv_heads, uint32_t head_dim, float scale,
    uint32_t excl_lo_end, uint32_t excl_hi_begin, uint32_t n_subblocks,
    uint32_t reduce_max, uint32_t accumulate, cudaStream_t stream);
bool launch_block_attn_score_softmax_pages_typed(
    float *score, const void *q_multi, bool query_is_fp16,
    const void *kbar_multi, KbarDType kbar_dtype,
    uint32_t n_layers, uint32_t n_tokens,
    uint32_t q_layer_stride, uint32_t n_blocks,
    uint32_t kbar_layer_stride, uint32_t n_heads,
    uint32_t n_kv_heads, uint32_t head_dim, float scale,
    uint32_t excl_lo_end, uint32_t excl_hi_begin, uint32_t n_subblocks,
    uint32_t reduce_max, uint32_t accumulate, cudaStream_t stream);
bool launch_block_attn_softmax_pages_stream_lse_typed(
    float *global_max, float *global_sum,
    const void *q_multi, bool query_is_fp16,
    const __half *kbar_tile,
    uint32_t n_layers, uint32_t n_tokens,
    uint32_t q_layer_stride, uint32_t tile_blocks,
    uint32_t kbar_layer_stride, uint32_t global_block_base,
    uint32_t global_n_blocks, uint32_t n_heads,
    uint32_t n_kv_heads, uint32_t head_dim, float scale,
    uint32_t excl_lo_end, uint32_t excl_hi_begin,
    uint32_t n_subblocks, uint32_t initialize,
    cudaStream_t stream);
bool launch_block_attn_softmax_pages_stream_score_typed(
    float *score, const void *q_multi, bool query_is_fp16,
    const __half *kbar_tile,
    const float *global_max, const float *global_sum,
    uint32_t n_layers, uint32_t n_tokens,
    uint32_t q_layer_stride, uint32_t tile_blocks,
    uint32_t kbar_layer_stride, uint32_t global_block_base,
    uint32_t global_n_blocks, uint32_t n_heads,
    uint32_t n_kv_heads, uint32_t head_dim, float scale,
    uint32_t excl_lo_end, uint32_t excl_hi_begin,
    uint32_t n_subblocks, uint32_t reduce_max,
    cudaStream_t stream);
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

template <typename T>
__global__ void quantize_kbar(const float *src, T *dst, uint64_t count) {
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
         i < count;
         i += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        dst[i] = T(src[i]);
    }
}

template <typename T>
__global__ void dequantize_kbar(const T *src, float *dst, uint64_t count) {
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
         i < count;
         i += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        dst[i] = static_cast<float>(src[i]);
    }
}

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
        bool force_tiled,
        bool force_two_dot = false) {
    if (force_tiled)
        setenv("QW3_KVMEM_SOFTMAX_PAGES_FORCE_TILED", "1", 1);
    else
        unsetenv("QW3_KVMEM_SOFTMAX_PAGES_FORCE_TILED");
    if (force_two_dot)
        setenv("QW3_KVMEM_SOFTMAX_PAGES_SCALABLE", "two_dot", 1);
    else
        unsetenv("QW3_KVMEM_SOFTMAX_PAGES_SCALABLE");
    if (!qw3::ported::launch_block_attn_score_softmax_pages(
            d_score, d_q, d_kbar, layers, tokens, q_stride, blocks,
            kbar_stride, heads, kv_heads, dim, scale, tc.excl_lo, tc.excl_hi,
            tc.subblocks, tc.reduce_max ? 1u : 0u, /*accumulate=*/0, 0)) {
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

template <typename T>
static std::vector<float> run_gpu_typed(
        float *d_score,
        const float *d_q,
        const float *d_kbar,
        uint64_t k_count,
        qw3::ported::KbarDType dtype,
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
        std::vector<float> *quantized_host) {
    T *d_typed = nullptr;
    float *d_dequant = nullptr;
    CHECK(cudaMalloc(&d_typed, k_count * sizeof(T)));
    CHECK(cudaMalloc(&d_dequant, k_count * sizeof(float)));
    const uint32_t threads = 256;
    const uint32_t grid = static_cast<uint32_t>(
        std::min<uint64_t>((k_count + threads - 1) / threads, 65535));
    quantize_kbar<<<grid, threads>>>(d_kbar, d_typed, k_count);
    dequantize_kbar<<<grid, threads>>>(d_typed, d_dequant, k_count);
    CHECK(cudaGetLastError());
    quantized_host->resize(k_count);
    CHECK(cudaMemcpy(quantized_host->data(), d_dequant,
                     k_count * sizeof(float), cudaMemcpyDeviceToHost));
    if (!qw3::ported::launch_block_attn_score_softmax_pages_typed(
            d_score, d_q, /*query_is_fp16=*/false, d_typed, dtype,
            layers, tokens, q_stride, blocks,
            kbar_stride, heads, kv_heads, dim, scale, tc.excl_lo, tc.excl_hi,
            tc.subblocks, tc.reduce_max ? 1u : 0u,
            /*accumulate=*/0, /*stream=*/0)) {
        std::fprintf(stderr, "typed launcher rejected blocks=%u subblocks=%u\n",
                     blocks, tc.subblocks);
        std::exit(1);
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<float> out(blocks);
    CHECK(cudaMemcpy(out.data(), d_score, blocks * sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_dequant));
    CHECK(cudaFree(d_typed));
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
    const std::vector<float> one_dot = run_gpu(
        d_score, d_q, d_kbar, layers, tokens, q_stride, tc.blocks,
        kbar_stride, heads, kv_heads, dim, scale, tc, true);
    char label[128];
    std::snprintf(label, sizeof(label), "one-dot-vs-host B=%u ns=%u max=%u",
                  tc.blocks, tc.subblocks, tc.reduce_max ? 1u : 0u);
    compare(label, one_dot, ref, 3e-6, 3e-3);

    const std::vector<float> two_dot = run_gpu(
        d_score, d_q, d_kbar, layers, tokens, q_stride, tc.blocks,
        kbar_stride, heads, kv_heads, dim, scale, tc, true, true);
    std::snprintf(label, sizeof(label), "one-dot-vs-two-dot B=%u ns=%u max=%u",
                  tc.blocks, tc.subblocks, tc.reduce_max ? 1u : 0u);
    compare(label, one_dot, two_dot, 3e-6, 3e-3);
    unsetenv("QW3_KVMEM_SOFTMAX_PAGES_SCALABLE");

    if (static_cast<uint64_t>(tc.blocks) * tc.subblocks <= 8192) {
        const std::vector<float> fused = run_gpu(
            d_score, d_q, d_kbar, layers, tokens, q_stride, tc.blocks,
            kbar_stride, heads, kv_heads, dim, scale, tc, false);
        std::snprintf(label, sizeof(label), "one-dot-vs-fused B=%u ns=%u max=%u",
                      tc.blocks, tc.subblocks, tc.reduce_max ? 1u : 0u);
        compare(label, one_dot, fused, 3e-6, 3e-3);
    }

    // Exercise the production low-precision index readers on both sides of the
    // fused/tiled dispatch boundary. The host reference uses the dequantized
    // stored values, isolating implementation correctness from expected
    // quantization error.
    if (tc.subblocks == 1 && (tc.blocks == 512 || tc.blocks == 8193)) {
        std::vector<float> qk;
        std::vector<float> f16 = run_gpu_typed<__half>(
            d_score, d_q, d_kbar, k_count, qw3::ported::KbarDType::F16,
            layers, tokens, q_stride, tc.blocks, kbar_stride, heads,
            kv_heads, dim, scale, tc, &qk);
        std::vector<float> qref = host_reference(
            q, qk, layers, tokens, q_stride, tc.blocks, kbar_stride, heads,
            kv_heads, dim, scale, tc.excl_lo, tc.excl_hi, tc.subblocks,
            tc.reduce_max);
        std::snprintf(label, sizeof(label), "fp16-vs-host B=%u", tc.blocks);
        compare(label, f16, qref, 3e-6, 3e-3);

        std::vector<float> fp8 = run_gpu_typed<__nv_fp8_e4m3>(
            d_score, d_q, d_kbar, k_count, qw3::ported::KbarDType::FP8,
            layers, tokens, q_stride, tc.blocks, kbar_stride, heads,
            kv_heads, dim, scale, tc, &qk);
        qref = host_reference(
            q, qk, layers, tokens, q_stride, tc.blocks, kbar_stride, heads,
            kv_heads, dim, scale, tc.excl_lo, tc.excl_hi, tc.subblocks,
            tc.reduce_max);
        std::snprintf(label, sizeof(label), "fp8-vs-host B=%u", tc.blocks);
        compare(label, fp8, qref, 3e-6, 3e-3);
    }

    CHECK(cudaFree(d_q));
    CHECK(cudaFree(d_kbar));
    CHECK(cudaFree(d_score));
}

static void run_fp16_query_chunk_test() {
    constexpr uint32_t layers = 2;
    constexpr uint32_t tokens = 7;
    constexpr uint32_t blocks = 8201;
    constexpr uint32_t heads = 4;
    constexpr uint32_t kv_heads = 2;
    constexpr uint32_t dim = 8;
    constexpr uint32_t split = 3;
    const uint64_t row = static_cast<uint64_t>(heads) * dim;
    const uint64_t q_count =
        static_cast<uint64_t>(layers) * tokens * row;
    const uint64_t k_count =
        static_cast<uint64_t>(layers) * blocks * kv_heads * dim;
    std::mt19937 rng(7781);
    std::normal_distribution<float> nd(0.0f, 0.35f);
    std::vector<float> q(q_count), kbar(k_count);
    std::vector<__half> q16(q_count);
    for (uint64_t i = 0; i < q_count; ++i) {
        q[i] = nd(rng);
        q16[i] = __float2half(q[i]);
    }
    for (float &v : kbar) v = nd(rng);

    float *d_q = nullptr;
    __half *d_q16 = nullptr;
    float *d_kbar = nullptr;
    float *d_score = nullptr;
    CHECK(cudaMalloc(&d_q, q_count * sizeof(float)));
    CHECK(cudaMalloc(&d_q16, q_count * sizeof(__half)));
    CHECK(cudaMalloc(&d_kbar, k_count * sizeof(float)));
    CHECK(cudaMalloc(&d_score, blocks * sizeof(float)));
    CHECK(cudaMemcpy(d_q, q.data(), q_count * sizeof(float),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_q16, q16.data(), q_count * sizeof(__half),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_kbar, kbar.data(), k_count * sizeof(float),
                     cudaMemcpyHostToDevice));
    setenv("QW3_KVMEM_SOFTMAX_PAGES_FORCE_TILED", "1", 1);

    auto launch = [&](const void *query, bool query_fp16,
                      uint32_t count, uint32_t stride,
                      uint32_t accumulate) {
        if (!qw3::ported::launch_block_attn_score_softmax_pages_typed(
                d_score, query, query_fp16, d_kbar,
                qw3::ported::KbarDType::F32, layers, count, stride,
                blocks, blocks, heads, kv_heads, dim,
                1.0f / std::sqrt(static_cast<float>(dim)),
                /*excl_lo_end=*/8, /*excl_hi_begin=*/blocks,
                /*n_subblocks=*/1, /*reduce_max=*/0, accumulate,
                /*stream=*/0)) {
            std::fprintf(stderr, "FP16 query chunk launcher rejected\n");
            std::exit(1);
        }
    };
    launch(d_q, false, tokens, tokens, 0);
    CHECK(cudaDeviceSynchronize());
    std::vector<float> full32(blocks);
    CHECK(cudaMemcpy(full32.data(), d_score, blocks * sizeof(float),
                     cudaMemcpyDeviceToHost));

    launch(d_q16, true, tokens, tokens, 0);
    CHECK(cudaDeviceSynchronize());
    std::vector<float> full16(blocks);
    CHECK(cudaMemcpy(full16.data(), d_score, blocks * sizeof(float),
                     cudaMemcpyDeviceToHost));
    compare("fp16-query-vs-fp32-query", full16, full32, 3e-6, 5e-3);

    const uint32_t max_chunk = tokens - split;
    std::vector<__half> packed(
        static_cast<uint64_t>(layers) * max_chunk * row);
    __half *d_chunk = nullptr;
    CHECK(cudaMalloc(&d_chunk, packed.size() * sizeof(__half)));
    uint32_t chunk_index = 0;
    for (uint32_t begin : {0u, split}) {
        const uint32_t count =
            begin == 0 ? split : tokens - split;
        for (uint32_t layer = 0; layer < layers; ++layer) {
            std::memcpy(
                packed.data() + static_cast<uint64_t>(layer) * count * row,
                q16.data() +
                    (static_cast<uint64_t>(layer) * tokens + begin) * row,
                static_cast<size_t>(count * row) * sizeof(__half));
        }
        CHECK(cudaMemcpy(
            d_chunk, packed.data(),
            static_cast<size_t>(layers) * count * row * sizeof(__half),
            cudaMemcpyHostToDevice));
        launch(d_chunk, true, count, count,
               chunk_index == 0 ? 0u : 1u);
        ++chunk_index;
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<float> chunked(blocks);
    CHECK(cudaMemcpy(chunked.data(), d_score, blocks * sizeof(float),
                     cudaMemcpyDeviceToHost));
    compare("fp16-query-chunk-accumulate", chunked, full16, 3e-6, 3e-3);

    CHECK(cudaFree(d_chunk));
    CHECK(cudaFree(d_score));
    CHECK(cudaFree(d_kbar));
    CHECK(cudaFree(d_q16));
    CHECK(cudaFree(d_q));
}

static void run_host_index_stream_test() {
    constexpr uint32_t layers = 2;
    constexpr uint32_t tokens = 3;
    constexpr uint32_t blocks = 1301;
    constexpr uint32_t kbar_stride = blocks + 11;
    constexpr uint32_t tile_blocks = 512;
    constexpr uint32_t heads = 4;
    constexpr uint32_t kv_heads = 2;
    constexpr uint32_t dim = 8;
    constexpr uint32_t excl_lo = 7;
    constexpr uint32_t excl_hi = blocks - 13;
    constexpr uint32_t subblocks = 1;
    const uint64_t row = static_cast<uint64_t>(heads) * dim;
    const uint64_t block_elems =
        static_cast<uint64_t>(subblocks) * kv_heads * dim;
    const uint64_t q_count =
        static_cast<uint64_t>(layers) * tokens * row;
    const uint64_t k_count =
        static_cast<uint64_t>(layers) * kbar_stride * block_elems;
    const uint64_t stage_count =
        static_cast<uint64_t>(layers) * tile_blocks * block_elems;
    const uint64_t distributions =
        static_cast<uint64_t>(layers) * tokens * heads;

    std::mt19937 rng(90210);
    std::normal_distribution<float> nd(0.0f, 0.35f);
    std::vector<__half> q16(q_count), k16(k_count);
    std::vector<float> q(q_count), k(k_count);
    for (uint64_t i = 0; i < q_count; ++i) {
        q16[i] = __float2half(nd(rng));
        q[i] = __half2float(q16[i]);
    }
    for (uint64_t i = 0; i < k_count; ++i) {
        k16[i] = __float2half(nd(rng));
        k[i] = __half2float(k16[i]);
    }

    __half *d_q = nullptr;
    __half *d_k = nullptr;
    __half *d_stage = nullptr;
    float *d_score = nullptr;
    float *d_stream_score = nullptr;
    float *d_max = nullptr;
    float *d_sum = nullptr;
    CHECK(cudaMalloc(&d_q, q_count * sizeof(__half)));
    CHECK(cudaMalloc(&d_k, k_count * sizeof(__half)));
    CHECK(cudaMalloc(&d_stage, stage_count * sizeof(__half)));
    CHECK(cudaMalloc(&d_score, blocks * sizeof(float)));
    CHECK(cudaMalloc(&d_stream_score, blocks * sizeof(float)));
    CHECK(cudaMalloc(&d_max, distributions * sizeof(float)));
    CHECK(cudaMalloc(&d_sum, distributions * sizeof(float)));
    CHECK(cudaMemcpy(d_q, q16.data(), q_count * sizeof(__half),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_k, k16.data(), k_count * sizeof(__half),
                     cudaMemcpyHostToDevice));

    if (!qw3::ported::launch_block_attn_score_softmax_pages_typed(
            d_score, d_q, /*query_is_fp16=*/true, d_k,
            qw3::ported::KbarDType::F16, layers, tokens, tokens,
            blocks, kbar_stride, heads, kv_heads, dim,
            1.0f / std::sqrt(static_cast<float>(dim)),
            excl_lo, excl_hi, subblocks, /*reduce_max=*/0,
            /*accumulate=*/0, /*stream=*/0)) {
        std::fprintf(stderr, "full FP16 reference launcher rejected\n");
        std::exit(1);
    }
    CHECK(cudaMemset(d_stream_score, 0, blocks * sizeof(float)));

    std::vector<__half> packed(stage_count);
    auto upload_tile = [&](uint32_t base, uint32_t count) {
        for (uint32_t layer = 0; layer < layers; ++layer) {
            const uint64_t src =
                (static_cast<uint64_t>(layer) * kbar_stride + base) *
                block_elems;
            const uint64_t dst =
                static_cast<uint64_t>(layer) * tile_blocks * block_elems;
            std::memcpy(packed.data() + dst, k16.data() + src,
                        static_cast<size_t>(count * block_elems) *
                            sizeof(__half));
        }
        CHECK(cudaMemcpy(d_stage, packed.data(),
                         stage_count * sizeof(__half),
                         cudaMemcpyHostToDevice));
    };
    bool first = true;
    for (uint32_t base = 0; base < blocks; base += tile_blocks) {
        const uint32_t count =
            std::min<uint32_t>(tile_blocks, blocks - base);
        upload_tile(base, count);
        if (!qw3::ported::launch_block_attn_softmax_pages_stream_lse_typed(
                d_max, d_sum, d_q, /*query_is_fp16=*/true, d_stage,
                layers, tokens, tokens, count, tile_blocks, base, blocks,
                heads, kv_heads, dim,
                1.0f / std::sqrt(static_cast<float>(dim)),
                excl_lo, excl_hi, subblocks, first ? 1u : 0u, 0)) {
            std::fprintf(stderr, "stream LSE launcher rejected base=%u\n", base);
            std::exit(1);
        }
        CHECK(cudaDeviceSynchronize());
        first = false;
    }
    for (uint32_t base = 0; base < blocks; base += tile_blocks) {
        const uint32_t count =
            std::min<uint32_t>(tile_blocks, blocks - base);
        upload_tile(base, count);
        if (!qw3::ported::launch_block_attn_softmax_pages_stream_score_typed(
                d_stream_score, d_q, /*query_is_fp16=*/true, d_stage,
                d_max, d_sum, layers, tokens, tokens, count,
                tile_blocks, base, blocks, heads, kv_heads, dim,
                1.0f / std::sqrt(static_cast<float>(dim)),
                excl_lo, excl_hi, subblocks, /*reduce_max=*/0, 0)) {
            std::fprintf(stderr, "stream score launcher rejected base=%u\n", base);
            std::exit(1);
        }
        CHECK(cudaDeviceSynchronize());
    }

    std::vector<float> full(blocks), streamed(blocks);
    CHECK(cudaMemcpy(full.data(), d_score, blocks * sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(streamed.data(), d_stream_score,
                     blocks * sizeof(float), cudaMemcpyDeviceToHost));
    const std::vector<float> host = host_reference(
        q, k, layers, tokens, tokens, blocks, kbar_stride,
        heads, kv_heads, dim,
        1.0f / std::sqrt(static_cast<float>(dim)),
        excl_lo, excl_hi, subblocks, false);
    compare("host-index-stream-vs-full", streamed, full, 3e-6, 3e-3);
    compare("host-index-stream-vs-host", streamed, host, 3e-6, 3e-3);

    CHECK(cudaFree(d_sum));
    CHECK(cudaFree(d_max));
    CHECK(cudaFree(d_stream_score));
    CHECK(cudaFree(d_score));
    CHECK(cudaFree(d_stage));
    CHECK(cudaFree(d_k));
    CHECK(cudaFree(d_q));
}

static float benchmark_path(float *d_score,
                            const float *d_q,
                            const __half *d_kbar,
                            uint32_t layers,
                            uint32_t tokens,
                            uint32_t blocks,
                            uint32_t heads,
                            uint32_t kv_heads,
                            uint32_t dim,
                            bool two_dot) {
    setenv("QW3_KVMEM_SOFTMAX_PAGES_FORCE_TILED", "1", 1);
    if (two_dot)
        setenv("QW3_KVMEM_SOFTMAX_PAGES_SCALABLE", "two_dot", 1);
    else
        unsetenv("QW3_KVMEM_SOFTMAX_PAGES_SCALABLE");
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    CHECK(cudaEventCreate(&begin));
    CHECK(cudaEventCreate(&end));
    CHECK(cudaEventRecord(begin));
    if (!qw3::ported::launch_block_attn_score_softmax_pages_typed(
            d_score, d_q, /*query_is_fp16=*/false, d_kbar,
            qw3::ported::KbarDType::F16,
            layers, tokens, tokens, blocks, blocks, heads, kv_heads, dim,
            1.0f / std::sqrt(static_cast<float>(dim)),
            /*excl_lo_end=*/8, /*excl_hi_begin=*/blocks,
            /*n_subblocks=*/1, /*reduce_max=*/0, /*accumulate=*/0, 0)) {
        std::fprintf(stderr, "benchmark launcher rejected path=%s tokens=%u\n",
                     two_dot ? "two-dot" : "one-dot", tokens);
        std::exit(1);
    }
    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));
    float elapsed_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&elapsed_ms, begin, end));
    CHECK(cudaEventDestroy(begin));
    CHECK(cudaEventDestroy(end));
    return elapsed_ms;
}

static void run_benchmark() {
    // Mirrors the long LongMemEval-M outlier shape:
    //   16 normal-attention layers, 32 Q heads / 8 KV heads, dim 128,
    //   ~32.4K indexed 32-token blocks. Device inputs are zero-filled because
    //   scorer runtime depends on shape, not values.
    constexpr uint32_t layers = 16;
    constexpr uint32_t blocks = 32438;
    constexpr uint32_t heads = 32;
    constexpr uint32_t kv_heads = 8;
    constexpr uint32_t dim = 128;
    constexpr uint32_t max_tokens = 6905;
    const uint64_t q_count =
        static_cast<uint64_t>(layers) * max_tokens * heads * dim;
    const uint64_t k_count =
        static_cast<uint64_t>(layers) * blocks * kv_heads * dim;
    float *d_q = nullptr;
    __half *d_kbar = nullptr;
    float *d_score = nullptr;
    CHECK(cudaMalloc(&d_q, q_count * sizeof(float)));
    CHECK(cudaMalloc(&d_kbar, k_count * sizeof(__half)));
    CHECK(cudaMalloc(&d_score, blocks * sizeof(float)));
    CHECK(cudaMemset(d_q, 0, q_count * sizeof(float)));
    CHECK(cudaMemset(d_kbar, 0, k_count * sizeof(__half)));

    // Median/ordinary query and a sizeable long-query point get both paths.
    // The full 6905-token outlier runs one-dot only; its production two-dot
    // baseline is already recorded in the session profile (~145 s).
    for (uint32_t tokens : {45u, 512u}) {
        const float one_ms = benchmark_path(
            d_score, d_q, d_kbar, layers, tokens, blocks,
            heads, kv_heads, dim, false);
        const float two_ms = benchmark_path(
            d_score, d_q, d_kbar, layers, tokens, blocks,
            heads, kv_heads, dim, true);
        std::printf(
            "[kvmem-softmax-bench] tokens=%u blocks=%u one_dot_ms=%.3f "
            "two_dot_ms=%.3f speedup=%.3fx\n",
            tokens, blocks, one_ms, two_ms, two_ms / one_ms);
    }
    const float long_one_ms = benchmark_path(
        d_score, d_q, d_kbar, layers, max_tokens, blocks,
        heads, kv_heads, dim, false);
    std::printf(
        "[kvmem-softmax-bench] tokens=%u blocks=%u one_dot_ms=%.3f "
        "two_dot_ms=not-run production_two_dot_ms=145312 "
        "production_baseline_speedup=%.3fx\n",
        max_tokens, blocks, long_one_ms, 145312.0f / long_one_ms);

    unsetenv("QW3_KVMEM_SOFTMAX_PAGES_FORCE_TILED");
    unsetenv("QW3_KVMEM_SOFTMAX_PAGES_SCALABLE");
    CHECK(cudaFree(d_score));
    CHECK(cudaFree(d_kbar));
    CHECK(cudaFree(d_q));
}

int main(int argc, char **argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::fprintf(stderr, "[kvmem-softmax-pages] no CUDA device; skipping\n");
        return 0;
    }
    if (argc > 1 && std::strcmp(argv[1], "--benchmark") == 0) {
        run_benchmark();
        return 0;
    }
    run_case({512, 1, 0, UINT32_MAX, false});
    run_case({4096, 1, 0, UINT32_MAX, false});
    run_case({8192, 1, 0, UINT32_MAX, false});
    run_case({8193, 1, 0, UINT32_MAX, false});
    run_case({4100, 2, 3, 4095, false});
    run_case({4100, 2, 3, 4095, true});
    run_fp16_query_chunk_test();
    run_host_index_stream_test();
    unsetenv("QW3_KVMEM_SOFTMAX_PAGES_FORCE_TILED");
    unsetenv("QW3_KVMEM_SOFTMAX_PAGES_SCALABLE");
    std::printf("[kvmem-softmax-pages] PASS\n");
    return 0;
}
