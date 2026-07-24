// Immutable raw-K construction regression.
//
// A long transcript can reselect the same fp16 block thousands of times.  The
// legacy in-place recipe repeatedly applies inverse/forward RoPE to the rounded
// result. The raw-K recipe periodically rebuilds the single active GPU K from
// an unrotated CPU mirror. This test covers both the new raw scatter+RoPE
// primitive and the drift that motivates periodic refresh.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace qw3::ported {
bool launch_rope_block_remap_paged(void *cache, bool is_fp16,
                                   uint32_t n_tokens, uint32_t n_kv_heads,
                                   uint32_t per_pos_size, uint32_t head_dim,
                                   uint32_t rope_dim, uint32_t win_base,
                                   int32_t orig_base, int32_t new_base,
                                   const int32_t *page_indices,
                                   uint32_t page_size, float theta,
                                   cudaStream_t stream);
bool launch_rope_block_remap_paged_fp8(void *cache,
                                       uint32_t n_tokens,
                                       uint32_t n_kv_heads,
                                       uint32_t per_pos_size,
                                       uint32_t head_dim,
                                       uint32_t rope_dim,
                                       uint32_t win_base,
                                       int32_t orig_base,
                                       int32_t new_base,
                                       const int32_t *page_indices,
                                       uint32_t page_size,
                                       float theta,
                                       cudaStream_t stream);
bool launch_raw_k_scatter_rope_paged_batched(
        void *cache, const void *raw_k, bool is_fp16,
        uint64_t raw_element_offset, uint32_t n_blocks,
        uint32_t max_n_tokens, uint32_t n_kv_heads,
        uint32_t per_pos_size, uint32_t head_dim, uint32_t rope_dim,
        const int32_t *to_base, const int32_t *n_tokens,
        const int32_t *page_indices, uint32_t page_size, float theta,
        cudaStream_t stream);
bool launch_build_rope_sincos_table(
        float *table, uint32_t positions, uint32_t rope_dim, float theta,
        cudaStream_t stream);
bool launch_raw_k_scatter_rope_paged_batched_table(
        void *cache, const void *raw_k, bool is_fp16,
        uint64_t raw_element_offset, uint32_t n_blocks,
        uint32_t max_n_tokens, uint32_t n_kv_heads,
        uint32_t per_pos_size, uint32_t head_dim, uint32_t rope_dim,
        const int32_t *to_base, const int32_t *n_tokens,
        const int32_t *page_indices, uint32_t page_size,
        const float *rope_sincos, uint32_t rope_table_positions,
        cudaStream_t stream);
}

#define CUDA_CHECK(call) do {                                             \
    const cudaError_t err_ = (call);                                      \
    if (err_ != cudaSuccess) {                                            \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(err_));                           \
        std::exit(1);                                                     \
    }                                                                     \
} while (0)

__global__ void bake_to_half(const float *raw, __half *out,
                             uint32_t rows, uint32_t heads,
                             uint32_t head_dim, uint32_t rope_dim,
                             int32_t base_pos, float theta) {
    const uint32_t row = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t i = threadIdx.x;
    const uint32_t half = rope_dim / 2;
    if (row >= rows || head >= heads || i >= head_dim) return;
    const uint64_t off =
        (static_cast<uint64_t>(row) * heads + head) * head_dim;
    float v = raw[off + i];
    if (i < rope_dim) {
        const uint32_t pair = i < half ? i : i - half;
        const float x0 = raw[off + pair];
        const float x1 = raw[off + pair + half];
        const float inv = __powf(theta, -2.0f * float(pair) / float(rope_dim));
        float s, c;
        __sincosf(float(base_pos + int32_t(row)) * inv, &s, &c);
        v = i < half ? x0 * c - x1 * s : x0 * s + x1 * c;
    }
    out[off + i] = __float2half(v);
}

__global__ void bake_to_fp8(const float *raw, __nv_fp8_e4m3 *out,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim, uint32_t rope_dim,
                            int32_t base_pos, float theta) {
    const uint32_t row = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t i = threadIdx.x;
    const uint32_t half = rope_dim / 2;
    if (row >= rows || head >= heads || i >= head_dim) return;
    const uint64_t off =
        (static_cast<uint64_t>(row) * heads + head) * head_dim;
    float v = raw[off + i];
    if (i < rope_dim) {
        const uint32_t pair = i < half ? i : i - half;
        const float x0 = raw[off + pair];
        const float x1 = raw[off + pair + half];
        const float inv = __powf(theta, -2.0f * float(pair) / float(rope_dim));
        float s, c;
        __sincosf(float(base_pos + int32_t(row)) * inv, &s, &c);
        v = i < half ? x0 * c - x1 * s : x0 * s + x1 * c;
    }
    out[off + i] = __nv_fp8_e4m3(v);
}

__global__ void bake_half_to_half(const __half *raw, __half *out,
                                  uint32_t rows, uint32_t heads,
                                  uint32_t head_dim, uint32_t rope_dim,
                                  int32_t base_pos, float theta) {
    const uint32_t row = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t i = threadIdx.x;
    const uint32_t half = rope_dim / 2;
    if (row >= rows || head >= heads || i >= head_dim) return;
    const uint64_t off =
        (static_cast<uint64_t>(row) * heads + head) * head_dim;
    float v = __half2float(raw[off + i]);
    if (i < rope_dim) {
        const uint32_t pair = i < half ? i : i - half;
        const float x0 = __half2float(raw[off + pair]);
        const float x1 = __half2float(raw[off + pair + half]);
        const float inv =
            __powf(theta, -2.0f * float(pair) / float(rope_dim));
        float s, c;
        __sincosf(float(base_pos + int32_t(row)) * inv, &s, &c);
        v = i < half ? x0 * c - x1 * s : x0 * s + x1 * c;
    }
    out[off + i] = __float2half(v);
}

static void remap(__half *cache, int32_t from, int32_t to,
                  const int32_t *pages) {
    constexpr uint32_t rows = 32;
    constexpr uint32_t heads = 4;
    constexpr uint32_t head_dim = 256;
    constexpr uint32_t rope_dim = 128;
    constexpr uint32_t page_size = 16;
    constexpr float theta = 1.0e7f;
    if (!qw3::ported::launch_rope_block_remap_paged(
            cache, true, rows, heads, heads * head_dim, head_dim, rope_dim,
            /*win_base=*/0, from, to, pages, page_size, theta, 0)) {
        std::fprintf(stderr, "RoPE remap launcher rejected test input\n");
        std::exit(1);
    }
}

static void remap_fp8(__nv_fp8_e4m3 *cache, int32_t from, int32_t to,
                      const int32_t *pages) {
    constexpr uint32_t rows = 32;
    constexpr uint32_t heads = 4;
    constexpr uint32_t head_dim = 256;
    constexpr uint32_t rope_dim = 128;
    constexpr uint32_t page_size = 16;
    constexpr float theta = 1.0e7f;
    if (!qw3::ported::launch_rope_block_remap_paged_fp8(
            cache, rows, heads, heads * head_dim, head_dim, rope_dim,
            /*win_base=*/0, from, to, pages, page_size, theta, 0)) {
        std::fprintf(stderr, "FP8 RoPE remap launcher rejected test input\n");
        std::exit(1);
    }
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("[kvmem-immutable-k] no CUDA device; SKIP");
        return 0;
    }

    constexpr uint32_t rows = 32;
    constexpr uint32_t heads = 4;
    constexpr uint32_t head_dim = 256;
    constexpr uint32_t rope_dim = 128;
    constexpr int32_t source_base = 777777;
    constexpr float theta = 1.0e7f;
    constexpr size_t count = size_t(rows) * heads * head_dim;
    constexpr size_t bytes = count * sizeof(__half);

    std::mt19937 rng(0x1A2B3C4Du);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> raw(count);
    for (float &v : raw) v = dist(rng);

    float *d_raw = nullptr;
    __half *d_source = nullptr, *d_work = nullptr;
    __half *d_reference = nullptr, *d_legacy = nullptr;
    int32_t *d_pages = nullptr;
    CUDA_CHECK(cudaMalloc(&d_raw, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_source, bytes));
    CUDA_CHECK(cudaMalloc(&d_work, bytes));
    CUDA_CHECK(cudaMalloc(&d_reference, bytes));
    CUDA_CHECK(cudaMalloc(&d_legacy, bytes));
    CUDA_CHECK(cudaMalloc(&d_pages, 2 * sizeof(int32_t)));
    const int32_t pages[2] = {0, 1};
    CUDA_CHECK(cudaMemcpy(d_raw, raw.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pages, pages, sizeof(pages),
                          cudaMemcpyHostToDevice));

    // Packed unrotated fp16 raw K -> arbitrary paged window slot + one RoPE.
    // The result must match a direct bake from the same rounded raw values.
    __half *d_raw_half = nullptr, *d_scatter_cache = nullptr;
    __half *d_scatter_table_cache = nullptr;
    __half *d_scatter_ref = nullptr;
    int32_t *d_pages4 = nullptr, *d_to = nullptr, *d_ntok = nullptr;
    CUDA_CHECK(cudaMalloc(&d_raw_half, bytes));
    CUDA_CHECK(cudaMalloc(&d_scatter_cache, bytes * 2));
    CUDA_CHECK(cudaMalloc(&d_scatter_table_cache, bytes * 2));
    CUDA_CHECK(cudaMalloc(&d_scatter_ref, bytes));
    CUDA_CHECK(cudaMalloc(&d_pages4, 4 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_to, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_ntok, sizeof(int32_t)));
    bake_to_half<<<dim3(rows, heads), head_dim>>>(
        d_raw, d_raw_half, rows, heads, head_dim, /*rope_dim=*/0,
        /*base_pos=*/0, theta);
    const int32_t pages4[4] = {0, 1, 2, 3};
    const int32_t scatter_to = 32;
    const int32_t scatter_ntok = rows;
    CUDA_CHECK(cudaMemcpy(d_pages4, pages4, sizeof(pages4),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_to, &scatter_to, sizeof(scatter_to),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ntok, &scatter_ntok, sizeof(scatter_ntok),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_scatter_cache, 0, bytes * 2));
    CUDA_CHECK(cudaMemset(d_scatter_table_cache, 0, bytes * 2));
    if (!qw3::ported::launch_raw_k_scatter_rope_paged_batched(
            d_scatter_cache, d_raw_half, true, 0, 1, rows, heads,
            heads * head_dim, head_dim, rope_dim, d_to, d_ntok, d_pages4,
            16, theta, 0)) {
        std::fprintf(stderr, "raw K scatter launcher rejected test input\n");
        return 1;
    }
    constexpr uint32_t rope_table_positions = scatter_to + rows;
    float *d_rope_table = nullptr;
    const uint64_t rope_table_floats =
        static_cast<uint64_t>(rope_table_positions) *
        (rope_dim / 2u) * 2u;
    CUDA_CHECK(cudaMalloc(
        &d_rope_table, rope_table_floats * sizeof(float)));
    if (!qw3::ported::launch_build_rope_sincos_table(
            d_rope_table, rope_table_positions, rope_dim, theta, 0) ||
        !qw3::ported::launch_raw_k_scatter_rope_paged_batched_table(
            d_scatter_table_cache, d_raw_half, true, 0, 1, rows, heads,
            heads * head_dim, head_dim, rope_dim, d_to, d_ntok, d_pages4,
            16, d_rope_table, rope_table_positions, 0)) {
        std::fprintf(
            stderr, "table raw K scatter launcher rejected test input\n");
        return 1;
    }
    bake_half_to_half<<<dim3(rows, heads), head_dim>>>(
        d_raw_half, d_scatter_ref, rows, heads, head_dim, rope_dim,
        scatter_to, theta);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__half> scatter_got(count), scatter_table_got(count);
    std::vector<__half> scatter_ref(count);
    CUDA_CHECK(cudaMemcpy(scatter_got.data(), d_scatter_cache + count,
                          bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scatter_ref.data(), d_scatter_ref, bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        scatter_table_got.data(), d_scatter_table_cache + count, bytes,
        cudaMemcpyDeviceToHost));
    if (std::memcmp(scatter_got.data(), scatter_ref.data(), bytes) != 0) {
        float max_abs = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            max_abs = std::max(
                max_abs,
                std::fabs(__half2float(scatter_got[i]) -
                          __half2float(scatter_ref[i])));
        }
        std::fprintf(stderr,
                     "FAIL raw scatter differs from direct bake max_abs=%.8f\n",
                     max_abs);
        return 1;
    }
    if (std::memcmp(
            scatter_table_got.data(), scatter_ref.data(), bytes) != 0) {
        float max_abs = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            max_abs = std::max(
                max_abs,
                std::fabs(__half2float(scatter_table_got[i]) -
                          __half2float(scatter_ref[i])));
        }
        std::fprintf(
            stderr,
            "FAIL table raw scatter differs from direct bake max_abs=%.8f\n",
            max_abs);
        return 1;
    }

    bake_to_half<<<dim3(rows, heads), head_dim>>>(
        d_raw, d_source, rows, heads, head_dim, rope_dim, source_base, theta);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    std::vector<__half> source_before(count);
    CUDA_CHECK(cudaMemcpy(source_before.data(), d_source, bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(d_legacy, d_source, bytes, cudaMemcpyDeviceToDevice));

    // Vary the selected window position for 1800 transcript turns.  The last
    // target is fixed so it can be compared with a one-shot reference.
    int32_t legacy_from = source_base;
    int32_t final_target = 0;
    for (int turn = 0; turn < 1800; ++turn) {
        final_target = (turn == 1799)
            ? 123456
            : ((turn % 3) == 0 ? 0 : ((turn % 3) == 1 ? 200000 : 31));

        // Production immutable recipe: full source->working reset, then one
        // transform from the source construction frame.
        CUDA_CHECK(cudaMemcpy(d_work, d_source, bytes, cudaMemcpyDeviceToDevice));
        remap(d_work, source_base, final_target, d_pages);

        // Legacy diagnostic: transform the already-rounded previous result.
        remap(d_legacy, legacy_from, final_target, d_pages);
        legacy_from = final_target;
    }
    CUDA_CHECK(cudaMemcpy(d_reference, d_source, bytes,
                          cudaMemcpyDeviceToDevice));
    remap(d_reference, source_base, final_target, d_pages);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    std::vector<__half> source_after(count), work(count), ref(count), legacy(count);
    CUDA_CHECK(cudaMemcpy(source_after.data(), d_source, bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(work.data(), d_work, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref.data(), d_reference, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(legacy.data(), d_legacy, bytes,
                          cudaMemcpyDeviceToHost));

    if (std::memcmp(source_before.data(), source_after.data(), bytes) != 0) {
        std::fprintf(stderr, "FAIL immutable source K was modified\n");
        return 1;
    }
    if (std::memcmp(work.data(), ref.data(), bytes) != 0) {
        std::fprintf(stderr,
                     "FAIL 1800-turn working K differs from one-shot materialization\n");
        return 1;
    }

    float legacy_max_abs = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        legacy_max_abs = std::max(
            legacy_max_abs,
            std::fabs(__half2float(legacy[i]) - __half2float(ref[i])));
    }
    std::printf("[kvmem-immutable-k] turns=1800 immutable_max_abs=0 "
                "legacy_max_abs=%.6f\n", legacy_max_abs);
    if (legacy_max_abs < 1.0e-2f) {
        std::fprintf(stderr,
                     "FAIL legacy control did not accumulate a measurable error\n");
        return 1;
    }

    // Practical immutable mode uses FP8 source+working K to halve both the
    // normal K/V pool and the extra working-K copy. Resetting working bytes
    // from source before every lossy FP8 remap must still make the result
    // exactly independent of the number of prior reselections.
    constexpr size_t fp8_bytes = count * sizeof(__nv_fp8_e4m3);
    __nv_fp8_e4m3 *d_source_fp8 = nullptr, *d_work_fp8 = nullptr;
    __nv_fp8_e4m3 *d_reference_fp8 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_source_fp8, fp8_bytes));
    CUDA_CHECK(cudaMalloc(&d_work_fp8, fp8_bytes));
    CUDA_CHECK(cudaMalloc(&d_reference_fp8, fp8_bytes));
    bake_to_fp8<<<dim3(rows, heads), head_dim>>>(
        d_raw, d_source_fp8, rows, heads, head_dim, rope_dim, source_base,
        theta);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> source_fp8_before(fp8_bytes);
    CUDA_CHECK(cudaMemcpy(source_fp8_before.data(), d_source_fp8, fp8_bytes,
                          cudaMemcpyDeviceToHost));
    for (int turn = 0; turn < 1800; ++turn) {
        const int32_t target = (turn == 1799)
            ? 123456
            : ((turn % 3) == 0 ? 0 : ((turn % 3) == 1 ? 200000 : 31));
        CUDA_CHECK(cudaMemcpy(d_work_fp8, d_source_fp8, fp8_bytes,
                              cudaMemcpyDeviceToDevice));
        remap_fp8(d_work_fp8, source_base, target, d_pages);
    }
    CUDA_CHECK(cudaMemcpy(d_reference_fp8, d_source_fp8, fp8_bytes,
                          cudaMemcpyDeviceToDevice));
    remap_fp8(d_reference_fp8, source_base, 123456, d_pages);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> source_fp8_after(fp8_bytes), work_fp8(fp8_bytes),
                         ref_fp8(fp8_bytes);
    CUDA_CHECK(cudaMemcpy(source_fp8_after.data(), d_source_fp8, fp8_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(work_fp8.data(), d_work_fp8, fp8_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref_fp8.data(), d_reference_fp8, fp8_bytes,
                          cudaMemcpyDeviceToHost));
    if (source_fp8_before != source_fp8_after || work_fp8 != ref_fp8) {
        std::fprintf(stderr,
                     "FAIL FP8 immutable reset is not one-shot deterministic\n");
        return 1;
    }
    cudaFree(d_reference_fp8);
    cudaFree(d_work_fp8);
    cudaFree(d_source_fp8);

    cudaFree(d_ntok);
    cudaFree(d_to);
    cudaFree(d_pages4);
    cudaFree(d_scatter_ref);
    cudaFree(d_rope_table);
    cudaFree(d_scatter_table_cache);
    cudaFree(d_scatter_cache);
    cudaFree(d_raw_half);
    cudaFree(d_pages);
    cudaFree(d_legacy);
    cudaFree(d_reference);
    cudaFree(d_work);
    cudaFree(d_source);
    cudaFree(d_raw);
    std::puts("[kvmem-immutable-k] PASS");
    return 0;
}
