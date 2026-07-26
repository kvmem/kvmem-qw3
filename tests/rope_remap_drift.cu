// Repeated re-RoPE numerical-drift diagnostic.
//
// This is intentionally a diagnostic executable rather than a pass/fail unit
// test. It applies the production paged/table re-RoPE kernels repeatedly to a
// Qwen3.6-shaped K block, then compares the result with rebuilding the same
// target position directly from the immutable raw-K authority. FP16 and raw
// E4M3 FP8 use their real cache conversion paths.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace qw3::ported {
bool launch_build_rope_sincos_table(
        float *table, uint32_t positions, uint32_t rope_dim, float theta,
        cudaStream_t stream);
bool launch_rope_block_remap_paged_batched_table(
        void *cache, bool is_fp16, uint32_t n_blocks,
        uint32_t max_n_tokens, uint32_t n_kv_heads,
        uint32_t per_pos_size, uint32_t head_dim, uint32_t rope_dim,
        const int32_t *to_base, const int32_t *from_base,
        const int32_t *n_tokens, const int32_t *page_indices,
        uint32_t page_size, const float *rope_sincos,
        uint32_t rope_table_positions, cudaStream_t stream);
bool launch_rope_block_remap_paged_batched_table_fp8(
        void *cache, uint32_t n_blocks, uint32_t max_n_tokens,
        uint32_t n_kv_heads, uint32_t per_pos_size, uint32_t head_dim,
        uint32_t rope_dim, const int32_t *to_base,
        const int32_t *from_base, const int32_t *n_tokens,
        const int32_t *page_indices, uint32_t page_size,
        const float *rope_sincos, uint32_t rope_table_positions,
        cudaStream_t stream);
bool launch_raw_k_scatter_rope_paged_batched_table(
        void *cache, const void *raw_k, bool is_fp16,
        uint64_t raw_element_offset, uint32_t n_blocks,
        uint32_t max_n_tokens, uint32_t n_kv_heads,
        uint32_t per_pos_size, uint32_t head_dim, uint32_t rope_dim,
        const int32_t *to_base, const int32_t *n_tokens,
        const int32_t *page_indices, uint32_t page_size,
        const float *rope_sincos, uint32_t rope_table_positions,
        uint64_t raw_block_stride_elements, cudaStream_t stream);
bool launch_raw_k_scatter_rope_paged_batched_table_fp8(
        void *cache, const void *raw_k, uint64_t raw_element_offset,
        uint32_t n_blocks, uint32_t max_n_tokens, uint32_t n_kv_heads,
        uint32_t per_pos_size, uint32_t head_dim, uint32_t rope_dim,
        const int32_t *to_base, const int32_t *n_tokens,
        const int32_t *page_indices, uint32_t page_size,
        const float *rope_sincos, uint32_t rope_table_positions,
        uint64_t raw_block_stride_elements, cudaStream_t stream);
}  // namespace qw3::ported

#define CUDA_CHECK(call) do {                                             \
    const cudaError_t err_ = (call);                                      \
    if (err_ != cudaSuccess) {                                            \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",                    \
                     __FILE__, __LINE__, cudaGetErrorString(err_));       \
        std::exit(1);                                                     \
    }                                                                     \
} while (0)

namespace {

constexpr uint32_t kTokens = 32;
constexpr uint32_t kKvHeads = 4;
constexpr uint32_t kHeadDim = 256;
constexpr uint32_t kRopeDim = 128;
constexpr uint32_t kPerPos = kKvHeads * kHeadDim;
constexpr uint32_t kPageSize = 16;
constexpr uint32_t kTablePositions = 204800;
constexpr float kTheta = 1.0e7f;
constexpr uint32_t kMaxRemaps = 128;

struct DeviceState {
    float *rope_table = nullptr;
    int32_t *page_indices = nullptr;
    int32_t *to_base = nullptr;
    int32_t *from_base = nullptr;
    int32_t *n_tokens = nullptr;

    ~DeviceState() {
        cudaFree(rope_table);
        cudaFree(page_indices);
        cudaFree(to_base);
        cudaFree(from_base);
        cudaFree(n_tokens);
    }
};

struct Metrics {
    double rel_l2 = 0.0;
    double one_minus_cos = 0.0;
    double max_abs_k = 0.0;
    double logit_rmse = 0.0;
    double logit_max_abs = 0.0;
    double softmax_tv_mean = 0.0;
    double softmax_max_abs = 0.0;
};

template <typename T>
float as_float(T value);

template <>
float as_float(__half value) {
    return __half2float(value);
}

template <>
float as_float(__nv_fp8_e4m3 value) {
    return static_cast<float>(value);
}

template <>
float as_float(float value) {
    return value;
}

template <typename T>
T from_float(float value);

template <>
__half from_float(float value) {
    return __float2half(value);
}

template <>
__nv_fp8_e4m3 from_float(float value) {
    return __nv_fp8_e4m3(value);
}

template <>
float from_float(float value) {
    return value;
}

template <typename T>
std::vector<float> copy_cache_to_float(const T *device_cache) {
    constexpr size_t n = static_cast<size_t>(kTokens) * kPerPos;
    std::vector<T> typed(n);
    CUDA_CHECK(cudaMemcpy(typed.data(), device_cache, n * sizeof(T),
                          cudaMemcpyDeviceToHost));
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = as_float(typed[i]);
    return out;
}

void rotate_query(std::vector<float> &q, uint32_t position) {
    const uint32_t half = kRopeDim / 2;
    for (uint32_t h = 0; h < kKvHeads; ++h) {
        float *head = q.data() + static_cast<size_t>(h) * kHeadDim;
        for (uint32_t i = 0; i < half; ++i) {
            const float inv = std::pow(
                kTheta, -2.0f * static_cast<float>(i) /
                            static_cast<float>(kRopeDim));
            const float angle = static_cast<float>(position) * inv;
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            const float x0 = head[i];
            const float x1 = head[i + half];
            head[i] = x0 * c - x1 * s;
            head[i + half] = x0 * s + x1 * c;
        }
    }
}

Metrics compare(const std::vector<float> &candidate,
                const std::vector<float> &reference,
                const std::vector<float> &query) {
    Metrics m;
    double err2 = 0.0;
    double ref2 = 0.0;
    double cand2 = 0.0;
    double dot = 0.0;
    for (size_t i = 0; i < candidate.size(); ++i) {
        const double c = candidate[i];
        const double r = reference[i];
        const double d = c - r;
        err2 += d * d;
        ref2 += r * r;
        cand2 += c * c;
        dot += c * r;
        m.max_abs_k = std::max(m.max_abs_k, std::abs(d));
    }
    m.rel_l2 = std::sqrt(err2 / std::max(ref2, 1.0e-30));
    m.one_minus_cos =
        1.0 - dot / std::sqrt(std::max(cand2 * ref2, 1.0e-30));

    const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
    double logit_err2 = 0.0;
    uint64_t logit_count = 0;
    for (uint32_t h = 0; h < kKvHeads; ++h) {
        std::vector<double> cand_logits(kTokens);
        std::vector<double> ref_logits(kTokens);
        for (uint32_t t = 0; t < kTokens; ++t) {
            const size_t koff =
                static_cast<size_t>(t) * kPerPos +
                static_cast<size_t>(h) * kHeadDim;
            const size_t qoff = static_cast<size_t>(h) * kHeadDim;
            double cs = 0.0;
            double rs = 0.0;
            for (uint32_t d = 0; d < kHeadDim; ++d) {
                cs += static_cast<double>(query[qoff + d]) *
                      candidate[koff + d];
                rs += static_cast<double>(query[qoff + d]) *
                      reference[koff + d];
            }
            cand_logits[t] = cs * scale;
            ref_logits[t] = rs * scale;
            const double diff = cand_logits[t] - ref_logits[t];
            logit_err2 += diff * diff;
            ++logit_count;
            m.logit_max_abs = std::max(m.logit_max_abs, std::abs(diff));
        }

        const double cmax =
            *std::max_element(cand_logits.begin(), cand_logits.end());
        const double rmax =
            *std::max_element(ref_logits.begin(), ref_logits.end());
        double csum = 0.0;
        double rsum = 0.0;
        for (uint32_t t = 0; t < kTokens; ++t) {
            cand_logits[t] = std::exp(cand_logits[t] - cmax);
            ref_logits[t] = std::exp(ref_logits[t] - rmax);
            csum += cand_logits[t];
            rsum += ref_logits[t];
        }
        double l1 = 0.0;
        for (uint32_t t = 0; t < kTokens; ++t) {
            const double cp = cand_logits[t] / csum;
            const double rp = ref_logits[t] / rsum;
            const double diff = std::abs(cp - rp);
            l1 += diff;
            m.softmax_max_abs = std::max(m.softmax_max_abs, diff);
        }
        m.softmax_tv_mean += 0.5 * l1 / kKvHeads;
    }
    m.logit_rmse =
        std::sqrt(logit_err2 / std::max<uint64_t>(logit_count, 1));
    return m;
}

std::vector<int32_t> make_positions(const std::string &pattern) {
    std::vector<int32_t> positions(kMaxRemaps + 1, 32768);
    if (pattern == "delta32") {
        for (uint32_t i = 1; i <= kMaxRemaps; ++i) {
            positions[i] = (i & 1) ? 32800 : 32768;
        }
    } else if (pattern == "delta4k") {
        for (uint32_t i = 1; i <= kMaxRemaps; ++i) {
            positions[i] = (i & 1) ? 36864 : 32768;
        }
    } else if (pattern == "delta64k") {
        for (uint32_t i = 1; i <= kMaxRemaps; ++i) {
            positions[i] = (i & 1) ? 98304 : 32768;
        }
    } else if (pattern == "random200k") {
        std::mt19937 rng(0x51a7u);
        std::uniform_int_distribution<int32_t> block(
            0, static_cast<int32_t>((kTablePositions - kTokens) / kTokens));
        for (uint32_t i = 1; i <= kMaxRemaps; ++i) {
            positions[i] = block(rng) * static_cast<int32_t>(kTokens);
        }
    } else {
        std::fprintf(stderr, "unknown position pattern: %s\n",
                     pattern.c_str());
        std::exit(2);
    }
    return positions;
}

template <typename T>
bool launch_fresh(T *cache, const T *raw, DeviceState &dev) {
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
        return qw3::ported::launch_raw_k_scatter_rope_paged_batched_table_fp8(
            cache, raw, 0, 1, kTokens, kKvHeads, kPerPos, kHeadDim,
            kRopeDim, dev.to_base, dev.n_tokens, dev.page_indices, kPageSize,
            dev.rope_table, kTablePositions,
            static_cast<uint64_t>(kTokens) * kPerPos, 0);
    } else {
        return qw3::ported::launch_raw_k_scatter_rope_paged_batched_table(
            cache, raw, std::is_same_v<T, __half>, 0, 1, kTokens, kKvHeads,
            kPerPos, kHeadDim, kRopeDim, dev.to_base, dev.n_tokens,
            dev.page_indices, kPageSize, dev.rope_table, kTablePositions,
            static_cast<uint64_t>(kTokens) * kPerPos, 0);
    }
}

template <typename T>
bool launch_remap(T *cache, DeviceState &dev) {
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
        return qw3::ported::launch_rope_block_remap_paged_batched_table_fp8(
            cache, 1, kTokens, kKvHeads, kPerPos, kHeadDim, kRopeDim,
            dev.to_base, dev.from_base, dev.n_tokens, dev.page_indices,
            kPageSize, dev.rope_table, kTablePositions, 0);
    } else {
        return qw3::ported::launch_rope_block_remap_paged_batched_table(
            cache, std::is_same_v<T, __half>, 1, kTokens, kKvHeads, kPerPos,
            kHeadDim, kRopeDim, dev.to_base, dev.from_base, dev.n_tokens,
            dev.page_indices, kPageSize, dev.rope_table, kTablePositions, 0);
    }
}

template <typename T>
void run_dtype(const char *dtype, DeviceState &dev,
               const std::vector<float> &raw_float,
               const std::vector<float> &query,
               const std::vector<float> &ideal_raw) {
    constexpr size_t n = static_cast<size_t>(kTokens) * kPerPos;
    std::vector<T> raw(n);
    for (size_t i = 0; i < n; ++i) raw[i] = from_float<T>(raw_float[i]);

    T *d_raw = nullptr;
    T *d_work = nullptr;
    T *d_fresh = nullptr;
    float *d_ideal_raw = nullptr;
    float *d_ideal_cache = nullptr;
    CUDA_CHECK(cudaMalloc(&d_raw, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_work, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_fresh, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_ideal_raw, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ideal_cache, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_raw, raw.data(), n * sizeof(T),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ideal_raw, ideal_raw.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));

    const std::vector<std::string> patterns = {
        "delta32", "delta4k", "delta64k", "random200k"};
    const std::vector<uint32_t> checkpoints = {
        0, 1, 2, 4, 8, 16, 32, 64, 128};

    for (const std::string &pattern : patterns) {
        const std::vector<int32_t> positions = make_positions(pattern);
        int32_t pos = positions[0];
        CUDA_CHECK(cudaMemcpy(dev.to_base, &pos, sizeof(pos),
                              cudaMemcpyHostToDevice));
        if (!launch_fresh(d_work, d_raw, dev)) {
            std::fprintf(stderr, "%s initial fresh launch failed\n", dtype);
            std::exit(1);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        uint64_t cumulative_delta = 0;
        size_t checkpoint_index = 0;
        for (uint32_t remaps = 0; remaps <= kMaxRemaps; ++remaps) {
            if (checkpoint_index < checkpoints.size() &&
                remaps == checkpoints[checkpoint_index]) {
                CUDA_CHECK(cudaMemcpy(dev.to_base, &pos, sizeof(pos),
                                      cudaMemcpyHostToDevice));
                if (!launch_fresh(d_fresh, d_raw, dev) ||
                    !qw3::ported::launch_raw_k_scatter_rope_paged_batched_table(
                        d_ideal_cache, d_ideal_raw, false, 0, 1, kTokens,
                        kKvHeads, kPerPos, kHeadDim, kRopeDim, dev.to_base,
                        dev.n_tokens, dev.page_indices, kPageSize,
                        dev.rope_table, kTablePositions,
                        static_cast<uint64_t>(kTokens) * kPerPos, 0)) {
                    std::fprintf(stderr, "%s reference launch failed\n",
                                 dtype);
                    std::exit(1);
                }
                CUDA_CHECK(cudaDeviceSynchronize());
                const std::vector<float> candidate =
                    copy_cache_to_float(d_work);
                const std::vector<float> fresh =
                    copy_cache_to_float(d_fresh);
                const std::vector<float> ideal =
                    copy_cache_to_float(d_ideal_cache);
                const Metrics drift = compare(candidate, fresh, query);
                const Metrics total = compare(candidate, ideal, query);
                std::printf(
                    "dtype=%-4s pattern=%-10s remaps=%3u "
                    "cum_delta=%9llu rel_l2=%.6g one_minus_cos=%.6g "
                    "max_abs_k=%.6g logit_rmse=%.6g logit_max=%.6g "
                    "softmax_tv=%.6g softmax_max=%.6g "
                    "total_rel_l2_vs_fp32=%.6g\n",
                    dtype, pattern.c_str(), remaps,
                    static_cast<unsigned long long>(cumulative_delta),
                    drift.rel_l2, drift.one_minus_cos, drift.max_abs_k,
                    drift.logit_rmse, drift.logit_max_abs,
                    drift.softmax_tv_mean, drift.softmax_max_abs,
                    total.rel_l2);
                ++checkpoint_index;
            }
            if (remaps == kMaxRemaps) break;
            const int32_t next = positions[remaps + 1];
            CUDA_CHECK(cudaMemcpy(dev.from_base, &pos, sizeof(pos),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dev.to_base, &next, sizeof(next),
                                  cudaMemcpyHostToDevice));
            if (!launch_remap(d_work, dev)) {
                std::fprintf(stderr, "%s remap launch failed\n", dtype);
                std::exit(1);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            cumulative_delta += static_cast<uint64_t>(
                std::llabs(static_cast<long long>(next) - pos));
            pos = next;
        }
    }

    cudaFree(d_raw);
    cudaFree(d_work);
    cudaFree(d_fresh);
    cudaFree(d_ideal_raw);
    cudaFree(d_ideal_cache);
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
        device_count == 0) {
        std::fprintf(stderr, "no CUDA devices, skipping\n");
        return 0;
    }

    DeviceState dev;
    const uint64_t table_floats =
        static_cast<uint64_t>(kTablePositions) * (kRopeDim / 2) * 2;
    CUDA_CHECK(cudaMalloc(&dev.rope_table, table_floats * sizeof(float)));
    if (!qw3::ported::launch_build_rope_sincos_table(
            dev.rope_table, kTablePositions, kRopeDim, kTheta, 0)) {
        std::fprintf(stderr, "RoPE table launch failed\n");
        return 1;
    }

    const uint32_t logical_pages =
        (kTablePositions + kPageSize - 1) / kPageSize;
    std::vector<int32_t> pages(logical_pages);
    for (uint32_t i = 0; i < logical_pages; ++i) {
        pages[i] = static_cast<int32_t>(i & 1u);
    }
    CUDA_CHECK(cudaMalloc(&dev.page_indices,
                          pages.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(dev.page_indices, pages.data(),
                          pages.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dev.to_base, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dev.from_base, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dev.n_tokens, sizeof(int32_t)));
    const int32_t n_tokens = kTokens;
    CUDA_CHECK(cudaMemcpy(dev.n_tokens, &n_tokens, sizeof(n_tokens),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    constexpr size_t n = static_cast<size_t>(kTokens) * kPerPos;
    std::mt19937 rng(0x726f7065u);
    std::normal_distribution<float> value_dist(0.0f, 1.0f);
    std::vector<float> raw(n);
    for (float &v : raw) v = value_dist(rng);
    std::vector<float> query(static_cast<size_t>(kKvHeads) * kHeadDim);
    for (float &v : query) v = value_dist(rng);
    rotate_query(query, kTablePositions - 1);

    std::printf(
        "# Qwen3.6 geometry: block=%u kv_heads=%u head_dim=%u "
        "rope_dim=%u theta=%.0f table_positions=%u\n",
        kTokens, kKvHeads, kHeadDim, kRopeDim, kTheta, kTablePositions);
    std::printf(
        "# rel_l2/logit/softmax fields compare repeated in-place re-RoPE "
        "against a canonical fresh rebuild in the same dtype.\n");
    std::printf(
        "# total_rel_l2_vs_fp32 also includes the dtype's one-time raw/cache "
        "quantization error.\n");

    run_dtype<__half>("fp16", dev, raw, query, raw);
    run_dtype<__nv_fp8_e4m3>("fp8", dev, raw, query, raw);
    return 0;
}
