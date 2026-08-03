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
bool launch_block_attn_score_softmax_groups_typed(
    float *score, const void *q_multi, bool query_is_fp16,
    const void *kbar_multi, KbarDType kbar_dtype,
    const int32_t *group_begin, const int32_t *group_end,
    uint32_t n_groups, uint32_t n_layers, uint32_t n_tokens,
    uint32_t q_layer_stride, uint32_t n_blocks,
    uint32_t kbar_layer_stride, uint32_t n_heads,
    uint32_t n_kv_heads, uint32_t head_dim, float scale,
    uint32_t excl_lo_end, uint32_t excl_hi_begin,
    uint32_t n_subblocks, uint32_t group_reduce_mass,
    uint32_t accumulate, cudaStream_t stream);
bool launch_block_attn_score_softmax_adaptive_typed(
    float *score, const void *q_multi, bool query_is_fp16,
    const void *packed_prototypes, KbarDType prototype_dtype,
    const int32_t *layer_offsets, const int32_t *block_offsets,
    const int32_t *block_counts, const int32_t *prototype_blocks,
    uint32_t n_layers, uint32_t n_tokens, uint32_t q_layer_stride,
    uint32_t n_blocks, uint32_t n_heads, uint32_t n_kv_heads,
    uint32_t head_dim, float scale, uint32_t excl_lo_end,
    uint32_t excl_hi_begin, uint32_t max_layer_prototypes,
    uint32_t accumulate, cudaStream_t stream);
bool launch_block_attn_stream_adaptive_lse_typed(
    float *global_max, float *global_sum,
    const void *q_multi, bool query_is_fp16,
    const void *prototype_tile, KbarDType prototype_dtype,
    uint32_t layer, uint32_t n_layers, uint32_t n_tokens,
    uint32_t q_layer_stride, uint32_t prototype_count,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
    float scale, uint32_t initialize, cudaStream_t stream);
bool launch_block_attn_stream_adaptive_score_typed(
    float *score, const void *q_multi, bool query_is_fp16,
    const void *prototype_tile, KbarDType prototype_dtype,
    const int32_t *block_offsets, const int32_t *block_counts,
    const float *global_max, const float *global_sum,
    uint32_t layer, uint32_t n_layers, uint32_t n_tokens,
    uint32_t q_layer_stride, uint32_t tile_blocks,
    uint32_t global_block_base, uint32_t n_heads,
    uint32_t n_kv_heads, uint32_t head_dim, float scale,
    cudaStream_t stream);
bool launch_block_attn_stream_adaptive_block_stats_typed(
    float *block_max, float *block_sum,
    const void *q_multi, bool query_is_fp16,
    const void *prototype_tile, KbarDType prototype_dtype,
    const int32_t *block_offsets, const int32_t *block_counts,
    uint32_t layer, uint32_t n_layers, uint32_t n_tokens,
    uint32_t q_layer_stride, uint32_t prototype_count,
    uint32_t tile_blocks, uint32_t workspace_block_base,
    uint32_t workspace_block_stride, uint32_t n_heads,
    uint32_t n_kv_heads, uint32_t head_dim, float scale,
    cudaStream_t stream);
bool launch_block_attn_stream_adaptive_finalize(
    float *score, const float *block_max, const float *block_sum,
    float *global_max, float *global_sum,
    uint32_t layer, uint32_t n_layers, uint32_t n_tokens,
    uint32_t workspace_block_stride, uint32_t block_count,
    uint32_t global_block_base, uint32_t n_heads,
    cudaStream_t stream);
bool launch_block_attn_score_adaptive_layer_typed(
    float *score, const void *q_multi, bool query_is_fp16,
    const void *prototype_layer, KbarDType prototype_dtype,
    const int32_t *block_offsets, const int32_t *block_counts,
    uint32_t layer, uint32_t n_layers, uint32_t n_tokens,
    uint32_t q_layer_stride, uint32_t prototype_count,
    uint32_t block_count, uint32_t global_block_base,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
    float scale, cudaStream_t stream);
bool launch_block_kdirection_adaptive_batch_typed(
    const float *k_batch, void *candidates, KbarDType candidate_dtype,
    float *residuals, uint32_t n_blocks_chunk, uint32_t k_stride,
    uint32_t batch, uint32_t blk_tokens, uint32_t n_kv_heads,
    uint32_t head_dim, uint32_t rope_dim, int32_t rope_base,
    float theta, cudaStream_t stream);
bool launch_block_attn_stream_lse_typed(
    float *global_max, float *global_sum,
    const void *q_multi, bool query_is_fp16,
    const void *kbar_tile, KbarDType kbar_dtype,
    uint32_t n_layers, uint32_t n_tokens, uint32_t q_layer_stride,
    uint32_t tile_blocks, uint32_t kbar_layer_stride,
    uint32_t global_block_base, uint32_t n_heads,
    uint32_t n_kv_heads, uint32_t head_dim, float scale,
    uint32_t excl_lo_end, uint32_t excl_hi_begin,
    uint32_t n_subblocks, uint32_t initialize, cudaStream_t stream);
bool launch_block_attn_stream_score_typed(
    float *score, const void *q_multi, bool query_is_fp16,
    const void *kbar_tile, KbarDType kbar_dtype,
    const float *global_max, const float *global_sum,
    uint32_t n_layers, uint32_t n_tokens, uint32_t q_layer_stride,
    uint32_t tile_blocks, uint32_t kbar_layer_stride,
    uint32_t global_block_base, uint32_t n_heads,
    uint32_t n_kv_heads, uint32_t head_dim, float scale,
    uint32_t excl_lo_end, uint32_t excl_hi_begin,
    uint32_t n_subblocks, uint32_t reduce_max,
    uint32_t accumulate, cudaStream_t stream);
bool launch_block_attn_stream_group_update_typed(
    float *group_dist, const void *q_multi, bool query_is_fp16,
    const void *kbar_tile, KbarDType kbar_dtype,
    const float *global_max, const float *global_sum,
    const int32_t *group_begin, const int32_t *group_end,
    uint32_t n_groups, uint32_t n_layers, uint32_t n_tokens,
    uint32_t q_layer_stride, uint32_t tile_blocks,
    uint32_t kbar_layer_stride, uint32_t global_block_base,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
    float scale, uint32_t excl_lo_end, uint32_t excl_hi_begin,
    uint32_t n_subblocks, uint32_t group_reduce_mass,
    cudaStream_t stream);
bool launch_block_attn_stream_group_finalize(
    float *score, const float *group_dist, uint32_t n_groups,
    uint32_t n_distributions, float distribution_weight,
    uint32_t accumulate, cudaStream_t stream);
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

static std::vector<float> host_adaptive_reference(
        const std::vector<float> &q,
        const std::vector<float> &prototypes,
        const std::vector<int32_t> &layer_offsets,
        const std::vector<int32_t> &block_offsets,
        const std::vector<int32_t> &block_counts,
        const std::vector<int32_t> &prototype_blocks,
        uint32_t layers, uint32_t tokens, uint32_t q_stride,
        uint32_t blocks, uint32_t heads, uint32_t kv_heads,
        uint32_t dim, float scale, uint32_t excl_lo,
        uint32_t excl_hi) {
    const uint32_t head_group = heads / kv_heads;
    std::vector<double> accum(blocks, 0.0);
    for (uint32_t l = 0; l < layers; ++l) {
        const uint32_t layer_begin =
            static_cast<uint32_t>(layer_offsets[l]);
        const uint32_t layer_end =
            static_cast<uint32_t>(layer_offsets[l + 1]);
        std::vector<double> logits(layer_end - layer_begin);
        for (uint32_t t = 0; t < tokens; ++t) {
            for (uint32_t h = 0; h < heads; ++h) {
                const uint32_t kh = h / head_group;
                double row_max = -INFINITY;
                for (uint32_t p = layer_begin; p < layer_end; ++p) {
                    const uint32_t w =
                        static_cast<uint32_t>(prototype_blocks[p]);
                    if (w < excl_lo || w >= excl_hi) {
                        logits[p - layer_begin] = -INFINITY;
                        continue;
                    }
                    double dot = 0.0;
                    for (uint32_t d = 0; d < dim; ++d) {
                        const uint64_t qi =
                            ((static_cast<uint64_t>(l) * q_stride + t) *
                                 heads + h) *
                                dim +
                            d;
                        const uint64_t ki =
                            (static_cast<uint64_t>(p) * kv_heads + kh) *
                                dim +
                            d;
                        dot += static_cast<double>(q[qi]) *
                               prototypes[ki];
                    }
                    logits[p - layer_begin] = dot * scale;
                    row_max =
                        std::max(row_max, logits[p - layer_begin]);
                }
                double denom = 0.0;
                for (double value : logits) {
                    if (std::isfinite(value)) {
                        denom += std::exp(value - row_max);
                    }
                }
                if (!(denom > 0.0)) continue;
                for (uint32_t w = excl_lo;
                     w < std::min(excl_hi, blocks); ++w) {
                    const uint32_t meta = l * blocks + w;
                    const uint32_t begin =
                        static_cast<uint32_t>(block_offsets[meta]);
                    const uint32_t count =
                        static_cast<uint32_t>(block_counts[meta]);
                    double block_max = 0.0;
                    for (uint32_t i = 0; i < count; ++i) {
                        const uint32_t p = begin + i;
                        block_max = std::max(
                            block_max,
                            std::exp(logits[p - layer_begin] - row_max) /
                                denom);
                    }
                    accum[w] += block_max;
                }
            }
        }
    }
    const double weight =
        1.0 / (static_cast<double>(layers) * heads);
    std::vector<float> out(blocks);
    for (uint32_t w = 0; w < blocks; ++w) {
        out[w] = static_cast<float>(accum[w] * weight);
    }
    return out;
}

static void run_adaptive_scorer_test() {
    constexpr uint32_t layers = 2;
    constexpr uint32_t tokens = 3;
    constexpr uint32_t q_stride = 5;
    constexpr uint32_t blocks = 73;
    // Match the production Qwen3.6-27B GQA ratio so packed, streamed, and
    // per-layer Adaptive scorers all exercise the group-6 specialization.
    constexpr uint32_t heads = 6;
    constexpr uint32_t kv_heads = 1;
    constexpr uint32_t dim = 8;
    constexpr uint32_t excl_lo = 3;
    constexpr uint32_t excl_hi = blocks - 4;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));

    std::vector<int32_t> layer_offsets(layers + 1, 0);
    std::vector<int32_t> block_offsets(layers * blocks, 0);
    std::vector<int32_t> block_counts(layers * blocks, 0);
    std::vector<int32_t> prototype_blocks;
    uint32_t max_layer_prototypes = 0;
    for (uint32_t l = 0; l < layers; ++l) {
        layer_offsets[l] =
            static_cast<int32_t>(prototype_blocks.size());
        const uint32_t layer_begin =
            static_cast<uint32_t>(prototype_blocks.size());
        for (uint32_t w = 0; w < blocks; ++w) {
            const uint32_t count =
                ((w + 2 * l) % 3 == 0) ? 1u
                : ((w + l) % 3 == 1) ? 2u
                                      : 4u;
            const uint32_t meta = l * blocks + w;
            block_offsets[meta] =
                static_cast<int32_t>(prototype_blocks.size());
            block_counts[meta] = static_cast<int32_t>(count);
            for (uint32_t p = 0; p < count; ++p) {
                prototype_blocks.push_back(static_cast<int32_t>(w));
            }
        }
        max_layer_prototypes = std::max(
            max_layer_prototypes,
            static_cast<uint32_t>(prototype_blocks.size()) -
                layer_begin);
    }
    layer_offsets[layers] =
        static_cast<int32_t>(prototype_blocks.size());

    const uint64_t q_count =
        static_cast<uint64_t>(layers) * q_stride * heads * dim;
    const uint64_t prototype_count =
        static_cast<uint64_t>(prototype_blocks.size()) *
        kv_heads * dim;
    std::mt19937 rng(44021);
    std::normal_distribution<float> nd(0.0f, 0.35f);
    std::vector<float> q(q_count), prototypes(prototype_count);
    for (float &v : q) v = nd(rng);
    for (float &v : prototypes) v = nd(rng);

    float *d_q = nullptr;
    float *d_prototypes = nullptr;
    float *d_score = nullptr;
    int32_t *d_layer_offsets = nullptr;
    int32_t *d_block_offsets = nullptr;
    int32_t *d_block_counts = nullptr;
    int32_t *d_prototype_blocks = nullptr;
    CHECK(cudaMalloc(&d_q, q_count * sizeof(float)));
    CHECK(cudaMalloc(
        &d_prototypes, prototype_count * sizeof(float)));
    CHECK(cudaMalloc(&d_score, blocks * sizeof(float)));
    CHECK(cudaMalloc(
        &d_layer_offsets, layer_offsets.size() * sizeof(int32_t)));
    CHECK(cudaMalloc(
        &d_block_offsets, block_offsets.size() * sizeof(int32_t)));
    CHECK(cudaMalloc(
        &d_block_counts, block_counts.size() * sizeof(int32_t)));
    CHECK(cudaMalloc(
        &d_prototype_blocks,
        prototype_blocks.size() * sizeof(int32_t)));
    CHECK(cudaMemcpy(
        d_q, q.data(), q_count * sizeof(float),
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        d_prototypes, prototypes.data(),
        prototype_count * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        d_layer_offsets, layer_offsets.data(),
        layer_offsets.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        d_block_offsets, block_offsets.data(),
        block_offsets.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        d_block_counts, block_counts.data(),
        block_counts.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        d_prototype_blocks, prototype_blocks.data(),
        prototype_blocks.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice));

    if (!qw3::ported::launch_block_attn_score_softmax_adaptive_typed(
            d_score, d_q, /*query_is_fp16=*/false,
            d_prototypes, qw3::ported::KbarDType::F32,
            d_layer_offsets, d_block_offsets, d_block_counts,
            d_prototype_blocks, layers, tokens, q_stride,
            blocks, heads, kv_heads, dim, scale, excl_lo, excl_hi,
            max_layer_prototypes, /*accumulate=*/0, /*stream=*/0)) {
        std::fprintf(stderr, "adaptive packed scorer rejected test\n");
        std::exit(1);
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<float> actual(blocks);
    CHECK(cudaMemcpy(
        actual.data(), d_score, blocks * sizeof(float),
        cudaMemcpyDeviceToHost));
    const std::vector<float> expected = host_adaptive_reference(
        q, prototypes, layer_offsets, block_offsets, block_counts,
        prototype_blocks, layers, tokens, q_stride, blocks, heads,
        kv_heads, dim, scale, excl_lo, excl_hi);
    compare(
        "adaptive-packed-vs-host", actual, expected, 3e-6, 3e-3);

    constexpr uint32_t tile_blocks = 11;
    float *d_global_max = nullptr;
    float *d_global_sum = nullptr;
    int32_t *d_tile_offsets = nullptr;
    int32_t *d_tile_counts = nullptr;
    const uint32_t distributions = layers * tokens * heads;
    CHECK(cudaMalloc(
        &d_global_max, distributions * sizeof(float)));
    CHECK(cudaMalloc(
        &d_global_sum, distributions * sizeof(float)));
    CHECK(cudaMalloc(
        &d_tile_offsets, blocks * sizeof(int32_t)));
    CHECK(cudaMalloc(
        &d_tile_counts, blocks * sizeof(int32_t)));
    CHECK(cudaMemset(d_score, 0, blocks * sizeof(float)));

    for (uint32_t l = 0; l < layers; ++l) {
        bool initialize = true;
        for (uint32_t first = excl_lo; first < excl_hi;
             first += tile_blocks) {
            const uint32_t count =
                std::min(tile_blocks, excl_hi - first);
            const uint32_t meta_begin = l * blocks + first;
            const uint32_t prototype_begin =
                static_cast<uint32_t>(block_offsets[meta_begin]);
            const uint32_t last_meta = meta_begin + count - 1;
            const uint32_t prototype_end =
                static_cast<uint32_t>(block_offsets[last_meta]) +
                static_cast<uint32_t>(block_counts[last_meta]);
            const uint32_t prototype_count =
                prototype_end - prototype_begin;
            const float *tile =
                d_prototypes +
                static_cast<uint64_t>(prototype_begin) *
                    kv_heads * dim;
            if (!qw3::ported::
                    launch_block_attn_stream_adaptive_lse_typed(
                        d_global_max, d_global_sum, d_q,
                        /*query_is_fp16=*/false, tile,
                        qw3::ported::KbarDType::F32, l, layers,
                        tokens, q_stride, prototype_count, heads,
                        kv_heads, dim, scale,
                        initialize ? 1u : 0u, /*stream=*/0)) {
                std::fprintf(
                    stderr,
                    "adaptive streamed LSE rejected test\n");
                std::exit(1);
            }
            initialize = false;
        }
    }
    for (uint32_t l = 0; l < layers; ++l) {
        for (uint32_t first = excl_lo; first < excl_hi;
             first += tile_blocks) {
            const uint32_t count =
                std::min(tile_blocks, excl_hi - first);
            const uint32_t meta_begin = l * blocks + first;
            const uint32_t prototype_begin =
                static_cast<uint32_t>(block_offsets[meta_begin]);
            const uint32_t last_meta = meta_begin + count - 1;
            const uint32_t prototype_end =
                static_cast<uint32_t>(block_offsets[last_meta]) +
                static_cast<uint32_t>(block_counts[last_meta]);
            const float *tile =
                d_prototypes +
                static_cast<uint64_t>(prototype_begin) *
                    kv_heads * dim;
            std::vector<int32_t> local_offsets(count);
            std::vector<int32_t> local_counts(count);
            for (uint32_t b = 0; b < count; ++b) {
                local_offsets[b] =
                    block_offsets[meta_begin + b] -
                    static_cast<int32_t>(prototype_begin);
                local_counts[b] = block_counts[meta_begin + b];
            }
            CHECK(cudaMemcpy(
                d_tile_offsets, local_offsets.data(),
                count * sizeof(int32_t), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(
                d_tile_counts, local_counts.data(),
                count * sizeof(int32_t), cudaMemcpyHostToDevice));
            if (!qw3::ported::
                    launch_block_attn_stream_adaptive_score_typed(
                        d_score, d_q, /*query_is_fp16=*/false,
                        tile, qw3::ported::KbarDType::F32,
                        d_tile_offsets, d_tile_counts,
                        d_global_max, d_global_sum, l, layers,
                        tokens, q_stride, count, first, heads,
                        kv_heads, dim, scale, /*stream=*/0)) {
                std::fprintf(
                    stderr,
                    "adaptive streamed score rejected test\n");
                std::exit(1);
            }
        }
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<float> streamed(blocks);
    CHECK(cudaMemcpy(
        streamed.data(), d_score, blocks * sizeof(float),
        cudaMemcpyDeviceToHost));
    compare(
        "adaptive-streamed-vs-packed", streamed, actual,
        3e-6, 3e-3);
    compare(
        "adaptive-streamed-vs-host", streamed, expected,
        3e-6, 3e-3);

    const uint32_t included_blocks = excl_hi - excl_lo;
    const uint64_t block_stats_entries =
        static_cast<uint64_t>(tokens) * heads * included_blocks;
    float *d_block_max = nullptr;
    float *d_block_sum = nullptr;
    CHECK(cudaMalloc(
        &d_block_max, block_stats_entries * sizeof(float)));
    CHECK(cudaMalloc(
        &d_block_sum, block_stats_entries * sizeof(float)));
    CHECK(cudaMemset(d_score, 0, blocks * sizeof(float)));
    for (uint32_t l = 0; l < layers; ++l) {
        for (uint32_t first = excl_lo; first < excl_hi;
             first += tile_blocks) {
            const uint32_t count =
                std::min(tile_blocks, excl_hi - first);
            const uint32_t meta_begin = l * blocks + first;
            const uint32_t prototype_begin =
                static_cast<uint32_t>(block_offsets[meta_begin]);
            const uint32_t last_meta = meta_begin + count - 1;
            const uint32_t prototype_end =
                static_cast<uint32_t>(block_offsets[last_meta]) +
                static_cast<uint32_t>(block_counts[last_meta]);
            const uint32_t prototype_count =
                prototype_end - prototype_begin;
            const float *tile =
                d_prototypes +
                static_cast<uint64_t>(prototype_begin) *
                    kv_heads * dim;
            std::vector<int32_t> local_offsets(count);
            std::vector<int32_t> local_counts(count);
            for (uint32_t b = 0; b < count; ++b) {
                local_offsets[b] =
                    block_offsets[meta_begin + b] -
                    static_cast<int32_t>(prototype_begin);
                local_counts[b] = block_counts[meta_begin + b];
            }
            CHECK(cudaMemcpy(
                d_tile_offsets, local_offsets.data(),
                count * sizeof(int32_t), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(
                d_tile_counts, local_counts.data(),
                count * sizeof(int32_t), cudaMemcpyHostToDevice));
            if (!qw3::ported::
                    launch_block_attn_stream_adaptive_block_stats_typed(
                        d_block_max, d_block_sum, d_q,
                        /*query_is_fp16=*/false, tile,
                        qw3::ported::KbarDType::F32,
                        d_tile_offsets, d_tile_counts, l, layers,
                        tokens, q_stride, prototype_count, count,
                        first - excl_lo, included_blocks, heads,
                        kv_heads, dim, scale, /*stream=*/0)) {
                std::fprintf(
                    stderr,
                    "adaptive tiled one-pass stats rejected test\n");
                std::exit(1);
            }
        }
        if (!qw3::ported::launch_block_attn_stream_adaptive_finalize(
                d_score, d_block_max, d_block_sum,
                d_global_max, d_global_sum, l, layers, tokens,
                included_blocks, included_blocks, excl_lo, heads,
                /*stream=*/0)) {
            std::fprintf(
                stderr,
                "adaptive tiled one-pass finalize rejected test\n");
            std::exit(1);
        }
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<float> tiled_one_pass(blocks);
    CHECK(cudaMemcpy(
        tiled_one_pass.data(), d_score, blocks * sizeof(float),
        cudaMemcpyDeviceToHost));
    compare(
        "adaptive-tiled-one-pass-vs-packed",
        tiled_one_pass, actual, 3e-6, 3e-3);
    compare(
        "adaptive-tiled-one-pass-vs-host",
        tiled_one_pass, expected, 3e-6, 3e-3);

    CHECK(cudaMemset(d_score, 0, blocks * sizeof(float)));
    for (uint32_t l = 0; l < layers; ++l) {
        const uint32_t meta_begin = l * blocks + excl_lo;
        const uint32_t prototype_begin =
            static_cast<uint32_t>(block_offsets[meta_begin]);
        const uint32_t last_meta =
            l * blocks + excl_hi - 1;
        const uint32_t prototype_end =
            static_cast<uint32_t>(block_offsets[last_meta]) +
            static_cast<uint32_t>(block_counts[last_meta]);
        const uint32_t layer_prototypes =
            prototype_end - prototype_begin;
        std::vector<int32_t> local_offsets(included_blocks);
        std::vector<int32_t> local_counts(included_blocks);
        for (uint32_t b = 0; b < included_blocks; ++b) {
            local_offsets[b] =
                block_offsets[meta_begin + b] -
                static_cast<int32_t>(prototype_begin);
            local_counts[b] = block_counts[meta_begin + b];
        }
        CHECK(cudaMemcpy(
            d_tile_offsets, local_offsets.data(),
            included_blocks * sizeof(int32_t),
            cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(
            d_tile_counts, local_counts.data(),
            included_blocks * sizeof(int32_t),
            cudaMemcpyHostToDevice));
        const float *layer_values =
            d_prototypes +
            static_cast<uint64_t>(prototype_begin) *
                kv_heads * dim;
        if (!qw3::ported::
                launch_block_attn_score_adaptive_layer_typed(
                    d_score, d_q, /*query_is_fp16=*/false,
                    layer_values, qw3::ported::KbarDType::F32,
                    d_tile_offsets, d_tile_counts, l, layers,
                    tokens, q_stride, layer_prototypes,
                    included_blocks, excl_lo, heads, kv_heads,
                    dim, scale, /*stream=*/0)) {
            std::fprintf(
                stderr,
                "adaptive layer one-pass scorer rejected test\n");
            std::exit(1);
        }
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<float> layer_one_pass(blocks);
    CHECK(cudaMemcpy(
        layer_one_pass.data(), d_score, blocks * sizeof(float),
        cudaMemcpyDeviceToHost));
    compare(
        "adaptive-layer-one-pass-vs-packed",
        layer_one_pass, actual, 3e-6, 3e-3);
    compare(
        "adaptive-layer-one-pass-vs-host",
        layer_one_pass, expected, 3e-6, 3e-3);

    CHECK(cudaFree(d_block_sum));
    CHECK(cudaFree(d_block_max));
    CHECK(cudaFree(d_tile_counts));
    CHECK(cudaFree(d_tile_offsets));
    CHECK(cudaFree(d_global_sum));
    CHECK(cudaFree(d_global_max));
    CHECK(cudaFree(d_prototype_blocks));
    CHECK(cudaFree(d_block_counts));
    CHECK(cudaFree(d_block_offsets));
    CHECK(cudaFree(d_layer_offsets));
    CHECK(cudaFree(d_score));
    CHECK(cudaFree(d_prototypes));
    CHECK(cudaFree(d_q));
}

static void run_adaptive_capture_test() {
    constexpr uint32_t block_tokens = 64;
    constexpr uint32_t batch = 83;
    constexpr uint32_t blocks = 2;
    constexpr uint32_t kv_heads = 2;
    constexpr uint32_t dim = 32;
    constexpr uint32_t slices = block_tokens / 32;
    constexpr uint32_t candidate_rows = blocks * slices * 7;
    constexpr uint32_t k_stride = kv_heads * dim;
    const uint64_t k_count =
        static_cast<uint64_t>(batch) * k_stride;
    const uint64_t candidate_count =
        static_cast<uint64_t>(candidate_rows) * kv_heads * dim;
    const uint64_t residual_count =
        static_cast<uint64_t>(blocks) * slices * kv_heads * 3;
    std::mt19937 rng(77403);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    std::vector<float> keys(k_count);
    for (float &v : keys) v = nd(rng);
    float *d_keys = nullptr;
    float *d_candidates = nullptr;
    float *d_residuals = nullptr;
    CHECK(cudaMalloc(&d_keys, k_count * sizeof(float)));
    CHECK(cudaMalloc(
        &d_candidates, candidate_count * sizeof(float)));
    CHECK(cudaMalloc(
        &d_residuals, residual_count * sizeof(float)));
    CHECK(cudaMemcpy(
        d_keys, keys.data(), k_count * sizeof(float),
        cudaMemcpyHostToDevice));
    if (!qw3::ported::launch_block_kdirection_adaptive_batch_typed(
            d_keys, d_candidates, qw3::ported::KbarDType::F32,
            d_residuals, blocks, k_stride, batch, block_tokens,
            kv_heads, dim, /*rope_dim=*/0, /*rope_base=*/0,
            /*theta=*/1000000.0f, /*stream=*/0)) {
        std::fprintf(stderr, "adaptive capture launcher rejected test\n");
        std::exit(1);
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<float> candidates(candidate_count);
    std::vector<float> residuals(residual_count);
    CHECK(cudaMemcpy(
        candidates.data(), d_candidates,
        candidate_count * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(
        residuals.data(), d_residuals,
        residual_count * sizeof(float), cudaMemcpyDeviceToHost));
    for (uint32_t job = 0; job < blocks * slices; ++job) {
        const uint32_t block = job / slices;
        const uint32_t slice = job % slices;
        const uint32_t row_begin =
            block * block_tokens + slice * 32;
        const uint32_t n =
            row_begin < batch ? std::min<uint32_t>(32, batch - row_begin)
                              : 0;
        for (uint32_t h = 0; h < kv_heads; ++h) {
            const uint64_t residual_base =
                (static_cast<uint64_t>(job) * kv_heads + h) * 3;
            if (n == 0) {
                if (residuals[residual_base] != -1.0f) {
                    std::fprintf(
                        stderr,
                        "FAIL adaptive capture missing slice was not marked\n");
                    std::exit(1);
                }
                continue;
            }
            const float e1 = residuals[residual_base + 0];
            const float e2 = residuals[residual_base + 1];
            const float e4 = residuals[residual_base + 2];
            if (!(e1 + 2e-6f >= e2 && e2 + 2e-6f >= e4 &&
                  e4 >= -2e-6f)) {
                std::fprintf(
                    stderr,
                    "FAIL adaptive residual monotonicity: %.8f %.8f %.8f\n",
                    e1, e2, e4);
                std::exit(1);
            }
            for (uint32_t d = 0; d < dim; ++d) {
                double mean = 0.0;
                for (uint32_t t = 0; t < n; ++t) {
                    mean += keys[
                        static_cast<uint64_t>(row_begin + t) * k_stride +
                        h * dim + d];
                }
                mean /= n;
                const uint64_t candidate =
                    (static_cast<uint64_t>(job) * 7 * kv_heads +
                     h) *
                        dim +
                    d;
                if (std::fabs(candidates[candidate] - mean) > 2e-6) {
                    std::fprintf(
                        stderr,
                        "FAIL adaptive P=1 mean mismatch job=%u h=%u d=%u\n",
                        job, h, d);
                    std::exit(1);
                }
            }
        }
    }
    std::printf(
        "[kvmem-softmax-pages] adaptive-capture residuals/means PASS\n");
    CHECK(cudaFree(d_residuals));
    CHECK(cudaFree(d_candidates));
    CHECK(cudaFree(d_keys));
}

static void run_streamed_index_test(bool reduce_max) {
    constexpr uint32_t layers = 2;
    constexpr uint32_t tokens = 5;
    constexpr uint32_t heads = 4;
    constexpr uint32_t kv_heads = 2;
    constexpr uint32_t dim = 8;
    constexpr uint32_t blocks = 257;
    constexpr uint32_t subblocks = 3;
    constexpr uint32_t tile_capacity = 61;
    constexpr uint32_t excl_lo = 2;
    constexpr uint32_t excl_hi = blocks - 3;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    const uint64_t q_count =
        static_cast<uint64_t>(layers) * tokens * heads * dim;
    const uint64_t block_elems =
        static_cast<uint64_t>(subblocks) * kv_heads * dim;
    const uint64_t k_count =
        static_cast<uint64_t>(layers) * blocks * block_elems;
    const uint32_t distributions = layers * tokens * heads;

    std::mt19937 rng(7713 + (reduce_max ? 1 : 0));
    std::normal_distribution<float> nd(0.0f, 0.35f);
    std::vector<__half> q(q_count), kbar(k_count);
    for (__half &value : q) value = __float2half(nd(rng));
    for (__half &value : kbar) value = __float2half(nd(rng));

    __half *d_q = nullptr;
    __half *d_kbar = nullptr;
    __half *d_tile = nullptr;
    float *d_resident = nullptr;
    float *d_streamed = nullptr;
    float *d_max = nullptr;
    float *d_sum = nullptr;
    CHECK(cudaMalloc(&d_q, q_count * sizeof(__half)));
    CHECK(cudaMalloc(&d_kbar, k_count * sizeof(__half)));
    CHECK(cudaMalloc(
        &d_tile, static_cast<uint64_t>(layers) * tile_capacity *
                     block_elems * sizeof(__half)));
    CHECK(cudaMalloc(&d_resident, blocks * sizeof(float)));
    CHECK(cudaMalloc(&d_streamed, blocks * sizeof(float)));
    CHECK(cudaMalloc(&d_max, distributions * sizeof(float)));
    CHECK(cudaMalloc(&d_sum, distributions * sizeof(float)));
    CHECK(cudaMemcpy(
        d_q, q.data(), q_count * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        d_kbar, kbar.data(), k_count * sizeof(__half),
        cudaMemcpyHostToDevice));

    if (!qw3::ported::launch_block_attn_score_softmax_pages_typed(
            d_resident, d_q, /*query_is_fp16=*/true, d_kbar,
            qw3::ported::KbarDType::F16, layers, tokens, tokens,
            blocks, blocks, heads, kv_heads, dim, scale,
            excl_lo, excl_hi, subblocks, reduce_max ? 1u : 0u,
            /*accumulate=*/0, /*stream=*/0)) {
        std::fprintf(stderr, "resident FP16 scorer rejected streamed test\n");
        std::exit(1);
    }

    std::vector<__half> packed(
        static_cast<uint64_t>(layers) * tile_capacity * block_elems);
    auto load_tile = [&](uint32_t first, uint32_t count) {
        for (uint32_t layer = 0; layer < layers; ++layer) {
            std::memcpy(
                packed.data() +
                    static_cast<uint64_t>(layer) * count * block_elems,
                kbar.data() +
                    (static_cast<uint64_t>(layer) * blocks + first) *
                        block_elems,
                static_cast<size_t>(count * block_elems) *
                    sizeof(__half));
        }
        CHECK(cudaMemcpy(
            d_tile, packed.data(),
            static_cast<size_t>(layers) * count * block_elems *
                sizeof(__half),
            cudaMemcpyHostToDevice));
    };
    const uint32_t tiles =
        (blocks + tile_capacity - 1) / tile_capacity;
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        const uint32_t first = tile * tile_capacity;
        const uint32_t count =
            std::min(tile_capacity, blocks - first);
        load_tile(first, count);
        if (!qw3::ported::launch_block_attn_stream_lse_typed(
                d_max, d_sum, d_q, /*query_is_fp16=*/true, d_tile,
                qw3::ported::KbarDType::F16, layers, tokens, tokens,
                count, count, first, heads, kv_heads, dim, scale,
                excl_lo, excl_hi, subblocks,
                /*initialize=*/tile == 0 ? 1u : 0u, /*stream=*/0)) {
            std::fprintf(stderr, "streamed LSE scorer rejected tile\n");
            std::exit(1);
        }
    }
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        const uint32_t first = tile * tile_capacity;
        const uint32_t count =
            std::min(tile_capacity, blocks - first);
        load_tile(first, count);
        if (!qw3::ported::launch_block_attn_stream_score_typed(
                d_streamed, d_q, /*query_is_fp16=*/true, d_tile,
                qw3::ported::KbarDType::F16, d_max, d_sum,
                layers, tokens, tokens, count, count, first,
                heads, kv_heads, dim, scale, excl_lo, excl_hi,
                subblocks, reduce_max ? 1u : 0u,
                /*accumulate=*/0, /*stream=*/0)) {
            std::fprintf(stderr, "streamed score scorer rejected tile\n");
            std::exit(1);
        }
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<float> resident(blocks), streamed(blocks);
    CHECK(cudaMemcpy(
        resident.data(), d_resident, blocks * sizeof(float),
        cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(
        streamed.data(), d_streamed, blocks * sizeof(float),
        cudaMemcpyDeviceToHost));
    compare(
        reduce_max ? "streamed-fp16-subblock-max"
                   : "streamed-fp16-subblock-sum",
        streamed, resident, 3e-6, 3e-3);

    // Existing semantic expansion can produce groups that cross staging-tile
    // boundaries. Verify both its max and mass reductions use the same global
    // softmax normalization as the resident scorer.
    const std::vector<int32_t> group_begin{
        static_cast<int32_t>(subblocks),
        static_cast<int32_t>(tile_capacity * subblocks - 7),
        static_cast<int32_t>(2 * tile_capacity * subblocks - 11),
        static_cast<int32_t>(4 * tile_capacity * subblocks - 5)};
    const std::vector<int32_t> group_end{
        static_cast<int32_t>(tile_capacity * subblocks + 9),
        static_cast<int32_t>(2 * tile_capacity * subblocks + 13),
        static_cast<int32_t>(4 * tile_capacity * subblocks + 17),
        static_cast<int32_t>(blocks * subblocks - subblocks)};
    const uint32_t groups =
        static_cast<uint32_t>(group_begin.size());
    int32_t *d_group_begin = nullptr;
    int32_t *d_group_end = nullptr;
    float *d_group_dist = nullptr;
    CHECK(cudaMalloc(&d_group_begin, groups * sizeof(int32_t)));
    CHECK(cudaMalloc(&d_group_end, groups * sizeof(int32_t)));
    CHECK(cudaMalloc(
        &d_group_dist,
        static_cast<uint64_t>(groups) * distributions * sizeof(float)));
    CHECK(cudaMemcpy(
        d_group_begin, group_begin.data(), groups * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        d_group_end, group_end.data(), groups * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    for (uint32_t reduce_mass : {0u, 1u}) {
        if (!qw3::ported::launch_block_attn_score_softmax_groups_typed(
                d_resident, d_q, /*query_is_fp16=*/true, d_kbar,
                qw3::ported::KbarDType::F16, d_group_begin, d_group_end,
                groups, layers, tokens, tokens, blocks, blocks,
                heads, kv_heads, dim, scale, excl_lo, excl_hi,
                subblocks, reduce_mass, /*accumulate=*/0, /*stream=*/0)) {
            std::fprintf(stderr, "resident group scorer rejected streamed test\n");
            std::exit(1);
        }
        CHECK(cudaMemset(
            d_group_dist, 0,
            static_cast<uint64_t>(groups) * distributions *
                sizeof(float)));
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            const uint32_t first = tile * tile_capacity;
            const uint32_t count =
                std::min(tile_capacity, blocks - first);
            load_tile(first, count);
            if (!qw3::ported::launch_block_attn_stream_group_update_typed(
                    d_group_dist, d_q, /*query_is_fp16=*/true, d_tile,
                    qw3::ported::KbarDType::F16, d_max, d_sum,
                    d_group_begin, d_group_end, groups, layers,
                    tokens, tokens, count, count, first, heads,
                    kv_heads, dim, scale, excl_lo, excl_hi,
                    subblocks, reduce_mass, /*stream=*/0)) {
                std::fprintf(
                    stderr, "streamed group scorer rejected tile\n");
                std::exit(1);
            }
        }
        if (!qw3::ported::launch_block_attn_stream_group_finalize(
                d_streamed, d_group_dist, groups, distributions,
                1.0f / static_cast<float>(layers * heads),
                /*accumulate=*/0, /*stream=*/0)) {
            std::fprintf(stderr, "streamed group finalize rejected\n");
            std::exit(1);
        }
        CHECK(cudaDeviceSynchronize());
        resident.resize(groups);
        streamed.resize(groups);
        CHECK(cudaMemcpy(
            resident.data(), d_resident, groups * sizeof(float),
            cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(
            streamed.data(), d_streamed, groups * sizeof(float),
            cudaMemcpyDeviceToHost));
        compare(
            reduce_mass ? "streamed-fp16-groups-mass"
                        : "streamed-fp16-groups-max",
            streamed, resident, 3e-6, 3e-3);
    }

    CHECK(cudaFree(d_group_dist));
    CHECK(cudaFree(d_group_end));
    CHECK(cudaFree(d_group_begin));
    CHECK(cudaFree(d_sum));
    CHECK(cudaFree(d_max));
    CHECK(cudaFree(d_streamed));
    CHECK(cudaFree(d_resident));
    CHECK(cudaFree(d_tile));
    CHECK(cudaFree(d_kbar));
    CHECK(cudaFree(d_q));
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

static std::vector<float> host_group_reference(
        const std::vector<float> &q,
        const std::vector<float> &kbar,
        const std::vector<int32_t> &group_begin,
        const std::vector<int32_t> &group_end,
        uint32_t layers, uint32_t tokens, uint32_t q_stride,
        uint32_t blocks, uint32_t kbar_stride, uint32_t heads,
        uint32_t kv_heads, uint32_t dim, float scale,
        uint32_t excl_lo, uint32_t excl_hi, uint32_t subblocks,
        bool reduce_mass) {
    const uint32_t total = blocks * subblocks;
    const uint32_t head_group = heads / kv_heads;
    std::vector<double> logits(total);
    std::vector<double> accum(group_begin.size(), 0.0);
    for (uint32_t l = 0; l < layers; ++l) {
        for (uint32_t t = 0; t < tokens; ++t) {
            for (uint32_t h = 0; h < heads; ++h) {
                const uint32_t kh = h / head_group;
                double row_max = -INFINITY;
                for (uint32_t p = 0; p < total; ++p) {
                    const uint32_t w = p / subblocks;
                    if (w < excl_lo || w >= excl_hi) {
                        logits[p] = -INFINITY;
                        continue;
                    }
                    double dot = 0.0;
                    for (uint32_t d = 0; d < dim; ++d) {
                        const uint64_t qi =
                            ((static_cast<uint64_t>(l) * q_stride + t) *
                                 heads + h) * dim + d;
                        const uint64_t ki =
                            (((static_cast<uint64_t>(l) * kbar_stride *
                                   subblocks + p) *
                                  kv_heads + kh) * dim) + d;
                        dot += static_cast<double>(q[qi]) * kbar[ki];
                    }
                    logits[p] = dot * scale;
                    row_max = std::max(row_max, logits[p]);
                }
                double sum = 0.0;
                for (double value : logits) {
                    if (std::isfinite(value)) {
                        sum += std::exp(value - row_max);
                    }
                }
                if (!(sum > 0.0)) continue;
                for (uint32_t g = 0; g < group_begin.size(); ++g) {
                    double mass = 0.0;
                    for (int32_t p = group_begin[g];
                         p < group_end[g]; ++p) {
                        if (p >= 0 &&
                            static_cast<uint32_t>(p) < total) {
                            const double probability =
                                std::exp(logits[p] - row_max) / sum;
                            mass = reduce_mass
                                ? mass + probability
                                : std::max(mass, probability);
                        }
                    }
                    accum[g] += mass;
                }
            }
        }
    }
    const double weight =
        1.0 / (static_cast<double>(layers) * heads);
    std::vector<float> result(accum.size());
    for (uint32_t g = 0; g < accum.size(); ++g) {
        result[g] = static_cast<float>(accum[g] * weight);
    }
    return result;
}

static void run_round_group_test(
        uint32_t blocks, uint32_t subblocks, bool reduce_mass) {
    constexpr uint32_t layers = 2;
    constexpr uint32_t tokens = 5;
    constexpr uint32_t q_stride = tokens;
    constexpr uint32_t heads = 4;
    constexpr uint32_t kv_heads = 2;
    constexpr uint32_t dim = 8;
    const uint32_t kbar_stride = blocks + 3;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    const uint32_t total = blocks * subblocks;
    const std::vector<int32_t> group_begin{
        static_cast<int32_t>(subblocks),
        static_cast<int32_t>(total / 5),
        static_cast<int32_t>(total / 2 - 1),
        static_cast<int32_t>(total * 3 / 4)};
    const std::vector<int32_t> group_end{
        static_cast<int32_t>(total / 5 + 3),
        static_cast<int32_t>(total / 2 + 2),
        static_cast<int32_t>(total * 3 / 4 + 5),
        static_cast<int32_t>(total - subblocks)};
    const uint64_t q_count =
        static_cast<uint64_t>(layers) * q_stride * heads * dim;
    const uint64_t k_count =
        static_cast<uint64_t>(layers) * kbar_stride * subblocks *
        kv_heads * dim;
    std::mt19937 rng(9917 + blocks + subblocks);
    std::normal_distribution<float> nd(0.0f, 0.35f);
    std::vector<float> q(q_count), kbar(k_count);
    for (float &value : q) value = nd(rng);
    for (float &value : kbar) value = nd(rng);

    float *d_q = nullptr;
    float *d_kbar = nullptr;
    float *d_score = nullptr;
    int32_t *d_group_begin = nullptr;
    int32_t *d_group_end = nullptr;
    CHECK(cudaMalloc(&d_q, q_count * sizeof(float)));
    CHECK(cudaMalloc(&d_kbar, k_count * sizeof(float)));
    CHECK(cudaMalloc(&d_score, blocks * sizeof(float)));
    CHECK(cudaMalloc(&d_group_begin,
                     group_begin.size() * sizeof(int32_t)));
    CHECK(cudaMalloc(&d_group_end,
                     group_end.size() * sizeof(int32_t)));
    CHECK(cudaMemcpy(d_q, q.data(), q_count * sizeof(float),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_kbar, kbar.data(), k_count * sizeof(float),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_group_begin, group_begin.data(),
                     group_begin.size() * sizeof(int32_t),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_group_end, group_end.data(),
                     group_end.size() * sizeof(int32_t),
                     cudaMemcpyHostToDevice));

    if (!qw3::ported::launch_block_attn_score_softmax_groups_typed(
            d_score, d_q, false, d_kbar,
            qw3::ported::KbarDType::F32,
            d_group_begin, d_group_end,
            static_cast<uint32_t>(group_begin.size()),
            layers, tokens, q_stride, blocks, kbar_stride,
            heads, kv_heads, dim, scale,
            /*excl_lo_end=*/1, /*excl_hi_begin=*/blocks - 1,
            subblocks, reduce_mass ? 1u : 0u,
            /*accumulate=*/0, /*stream=*/0)) {
        std::fprintf(stderr,
                     "round group launcher rejected B=%u ns=%u\n",
                     blocks, subblocks);
        std::exit(1);
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<float> actual(group_begin.size());
    CHECK(cudaMemcpy(actual.data(), d_score,
                     actual.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
    const std::vector<float> expected = host_group_reference(
        q, kbar, group_begin, group_end,
        layers, tokens, q_stride, blocks, kbar_stride,
        heads, kv_heads, dim, scale, 1, blocks - 1, subblocks,
        reduce_mass);
    char label[96];
    std::snprintf(label, sizeof(label),
                  "semantic-groups-vs-host B=%u ns=%u reduce=%s",
                  blocks, subblocks, reduce_mass ? "mass" : "max");
    compare(label, actual, expected, 3e-6, 3e-3);

    CHECK(cudaFree(d_group_end));
    CHECK(cudaFree(d_group_begin));
    CHECK(cudaFree(d_score));
    CHECK(cudaFree(d_kbar));
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

static float benchmark_adaptive_path(
        float *d_score,
        const __half *d_q,
        const __half *d_prototypes,
        const int32_t *d_layer_offsets,
        const int32_t *d_block_offsets,
        const int32_t *d_block_counts,
        const int32_t *d_prototype_blocks,
        uint32_t prototype_count,
        uint32_t max_layer_prototypes,
        uint32_t layers,
        uint32_t tokens,
        uint32_t blocks,
        uint32_t heads,
        uint32_t kv_heads,
        uint32_t dim,
        bool legacy_two_dot) {
    if (legacy_two_dot) {
        setenv("QW3_KVMEM_ADAPTIVE_SCORER", "two_dot", 1);
    } else {
        unsetenv("QW3_KVMEM_ADAPTIVE_SCORER");
    }
    CHECK(cudaMemset(d_score, 0, blocks * sizeof(float)));
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    CHECK(cudaEventCreate(&begin));
    CHECK(cudaEventCreate(&end));
    CHECK(cudaEventRecord(begin));
    if (!qw3::ported::launch_block_attn_score_softmax_adaptive_typed(
            d_score, d_q, /*query_is_fp16=*/true,
            d_prototypes, qw3::ported::KbarDType::F16,
            d_layer_offsets, d_block_offsets, d_block_counts,
            d_prototype_blocks, layers, tokens, tokens,
            blocks, heads, kv_heads, dim,
            1.0f / std::sqrt(static_cast<float>(dim)),
            /*excl_lo_end=*/8, /*excl_hi_begin=*/blocks,
            max_layer_prototypes, /*accumulate=*/0, /*stream=*/0)) {
        std::fprintf(
            stderr,
            "adaptive benchmark launcher rejected prototypes=%u "
            "tokens=%u legacy=%d\n",
            prototype_count, tokens, legacy_two_dot ? 1 : 0);
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

static void run_adaptive_benchmark() {
    // Mirrors the completed AgentLongBench-1M Adaptive run:
    //   16 normal-attention layers, 2027 logical 512-token blocks,
    //   16 independently clustered 32-token slices per block,
    //   24 Q heads / 8 KV heads / dim 128, and ~3.49 prototypes per slice.
    // The deterministic 83% P4 / 17% P1 mix approximates the observed
    // p1=87,202, p2=0, p4=429,726 distribution without retaining a multi-GiB
    // host-side prototype payload.
    constexpr uint32_t layers = 16;
    constexpr uint32_t blocks = 2027;
    constexpr uint32_t slices = 16;
    constexpr uint32_t heads = 24;
    constexpr uint32_t kv_heads = 8;
    constexpr uint32_t dim = 128;
    constexpr uint32_t tokens = 26;

    std::vector<int32_t> layer_offsets(layers + 1, 0);
    std::vector<int32_t> block_offsets(
        static_cast<uint64_t>(layers) * blocks, 0);
    std::vector<int32_t> block_counts(
        static_cast<uint64_t>(layers) * blocks, 0);
    std::vector<int32_t> prototype_blocks;
    uint64_t total_prototypes = 0;
    uint32_t max_layer_prototypes = 0;
    for (uint32_t l = 0; l < layers; ++l) {
        layer_offsets[l] = static_cast<int32_t>(total_prototypes);
        const uint64_t layer_begin = total_prototypes;
        for (uint32_t b = 0; b < blocks; ++b) {
            const uint64_t meta =
                static_cast<uint64_t>(l) * blocks + b;
            block_offsets[meta] =
                static_cast<int32_t>(total_prototypes);
            uint32_t count = 0;
            for (uint32_t s = 0; s < slices; ++s) {
                const uint32_t bucket =
                    (l * 37u + b * slices + s) % 100u;
                count += bucket < 83u ? 4u : 1u;
            }
            block_counts[meta] = static_cast<int32_t>(count);
            prototype_blocks.insert(
                prototype_blocks.end(), count,
                static_cast<int32_t>(b));
            total_prototypes += count;
        }
        max_layer_prototypes = std::max<uint32_t>(
            max_layer_prototypes,
            static_cast<uint32_t>(total_prototypes - layer_begin));
    }
    layer_offsets[layers] =
        static_cast<int32_t>(total_prototypes);
    if (total_prototypes > UINT32_MAX) {
        std::fprintf(stderr, "adaptive benchmark metadata overflow\n");
        std::exit(1);
    }

    const uint64_t q_count =
        static_cast<uint64_t>(layers) * tokens * heads * dim;
    const uint64_t prototype_elems =
        total_prototypes * kv_heads * dim;
    __half *d_q = nullptr;
    __half *d_prototypes = nullptr;
    float *d_score = nullptr;
    int32_t *d_layer_offsets = nullptr;
    int32_t *d_block_offsets = nullptr;
    int32_t *d_block_counts = nullptr;
    int32_t *d_prototype_blocks = nullptr;
    CHECK(cudaMalloc(&d_q, q_count * sizeof(__half)));
    CHECK(cudaMalloc(
        &d_prototypes, prototype_elems * sizeof(__half)));
    CHECK(cudaMalloc(&d_score, blocks * sizeof(float)));
    CHECK(cudaMalloc(
        &d_layer_offsets, layer_offsets.size() * sizeof(int32_t)));
    CHECK(cudaMalloc(
        &d_block_offsets, block_offsets.size() * sizeof(int32_t)));
    CHECK(cudaMalloc(
        &d_block_counts, block_counts.size() * sizeof(int32_t)));
    CHECK(cudaMalloc(
        &d_prototype_blocks,
        prototype_blocks.size() * sizeof(int32_t)));
    CHECK(cudaMemset(d_q, 0, q_count * sizeof(__half)));
    CHECK(cudaMemset(
        d_prototypes, 0, prototype_elems * sizeof(__half)));
    CHECK(cudaMemcpy(
        d_layer_offsets, layer_offsets.data(),
        layer_offsets.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        d_block_offsets, block_offsets.data(),
        block_offsets.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        d_block_counts, block_counts.data(),
        block_counts.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        d_prototype_blocks, prototype_blocks.data(),
        prototype_blocks.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice));

    // Warm CUDA's async allocation pool before the measured calls.
    (void)benchmark_adaptive_path(
        d_score, d_q, d_prototypes, d_layer_offsets,
        d_block_offsets, d_block_counts, d_prototype_blocks,
        static_cast<uint32_t>(total_prototypes),
        max_layer_prototypes, layers, tokens, blocks,
        heads, kv_heads, dim, /*legacy_two_dot=*/true);
    const float legacy_ms = benchmark_adaptive_path(
        d_score, d_q, d_prototypes, d_layer_offsets,
        d_block_offsets, d_block_counts, d_prototype_blocks,
        static_cast<uint32_t>(total_prototypes),
        max_layer_prototypes, layers, tokens, blocks,
        heads, kv_heads, dim, /*legacy_two_dot=*/true);
    setenv("QW3_KVMEM_ADAPTIVE_GQA", "0", 1);
    const float scalar_one_dot_ms = benchmark_adaptive_path(
        d_score, d_q, d_prototypes, d_layer_offsets,
        d_block_offsets, d_block_counts, d_prototype_blocks,
        static_cast<uint32_t>(total_prototypes),
        max_layer_prototypes, layers, tokens, blocks,
        heads, kv_heads, dim, /*legacy_two_dot=*/false);
    unsetenv("QW3_KVMEM_ADAPTIVE_GQA");
    const float default_ms = benchmark_adaptive_path(
        d_score, d_q, d_prototypes, d_layer_offsets,
        d_block_offsets, d_block_counts, d_prototype_blocks,
        static_cast<uint32_t>(total_prototypes),
        max_layer_prototypes, layers, tokens, blocks,
        heads, kv_heads, dim, /*legacy_two_dot=*/false);
    std::printf(
        "[kvmem-adaptive-bench] layers=%u tokens=%u blocks=%u "
        "prototypes=%llu avg_per_slice=%.3f index_gib=%.3f "
        "legacy_ms=%.3f scalar_one_dot_ms=%.3f "
        "gqa_one_dot_ms=%.3f legacy_speedup=%.3fx "
        "gqa_speedup=%.3fx\n",
        layers, tokens, blocks,
        static_cast<unsigned long long>(total_prototypes),
        static_cast<double>(total_prototypes) /
            (static_cast<double>(layers) * blocks * slices),
        static_cast<double>(prototype_elems * sizeof(__half)) /
            (1024.0 * 1024.0 * 1024.0),
        legacy_ms, scalar_one_dot_ms, default_ms,
        legacy_ms / default_ms,
        scalar_one_dot_ms / default_ms);

    unsetenv("QW3_KVMEM_ADAPTIVE_SCORER");
    unsetenv("QW3_KVMEM_ADAPTIVE_GQA");
    CHECK(cudaFree(d_prototype_blocks));
    CHECK(cudaFree(d_block_counts));
    CHECK(cudaFree(d_block_offsets));
    CHECK(cudaFree(d_layer_offsets));
    CHECK(cudaFree(d_score));
    CHECK(cudaFree(d_prototypes));
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
    if (argc > 1 &&
        std::strcmp(argv[1], "--benchmark-adaptive") == 0) {
        run_adaptive_benchmark();
        return 0;
    }
    run_case({512, 1, 0, UINT32_MAX, false});
    run_case({4096, 1, 0, UINT32_MAX, false});
    run_case({8192, 1, 0, UINT32_MAX, false});
    run_case({8193, 1, 0, UINT32_MAX, false});
    run_case({4100, 2, 3, 4095, false});
    run_case({4100, 2, 3, 4095, true});
    run_fp16_query_chunk_test();
    run_round_group_test(/*blocks=*/16, /*subblocks=*/4,
                         /*reduce_mass=*/false);
    run_round_group_test(/*blocks=*/16, /*subblocks=*/4,
                         /*reduce_mass=*/true);
    run_round_group_test(/*blocks=*/600, /*subblocks=*/16,
                         /*reduce_mass=*/false);
    run_round_group_test(/*blocks=*/600, /*subblocks=*/16,
                         /*reduce_mass=*/true);
    run_streamed_index_test(/*reduce_max=*/false);
    run_streamed_index_test(/*reduce_max=*/true);
    run_adaptive_capture_test();
    run_adaptive_scorer_test();
    unsetenv("QW3_KVMEM_SOFTMAX_PAGES_FORCE_TILED");
    unsetenv("QW3_KVMEM_SOFTMAX_PAGES_SCALABLE");
    std::printf("[kvmem-softmax-pages] PASS\n");
    return 0;
}
