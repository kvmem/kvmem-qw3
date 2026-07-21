// Immutable-source K construction regression.
//
// A long transcript can reselect the same fp16 block thousands of times.  The
// legacy in-place recipe repeatedly applies inverse/forward RoPE to the rounded
// result.  The immutable recipe copies source K into working K and applies
// exactly one source-frame -> window-frame transform on every assembly.  This
// test exercises 1800 changing selections and requires the final working bytes
// to be identical to a single direct materialization from source.

#include <cuda_fp16.h>
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

    cudaFree(d_pages);
    cudaFree(d_legacy);
    cudaFree(d_reference);
    cudaFree(d_work);
    cudaFree(d_source);
    cudaFree(d_raw);
    std::puts("[kvmem-immutable-k] PASS");
    return 0;
}
