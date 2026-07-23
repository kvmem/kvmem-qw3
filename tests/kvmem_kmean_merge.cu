// Incremental mean-K parity test.
//
// Verifies that a logical block built across arbitrary prefill boundaries has
// the same content-frame mean as one-shot capture, including sub-block mode and
// a block-aligned replay overwrite.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace qw3::ported {
enum class KbarDType : uint32_t {
    F32 = 0,
    F16 = 1,
    FP8 = 2,
};
bool launch_block_kmean_content_batch(
    const float *k_batch, float *kbar, uint64_t kbar_block_base,
    uint32_t n_blocks_chunk, uint32_t k_stride, uint32_t batch,
    uint32_t blk_tokens, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t rope_dim, int32_t rope_base, float theta,
    uint32_t n_subblocks, cudaStream_t stream);
bool launch_block_kmean_content_batch_merge(
    const float *k_batch, float *kbar, uint64_t kbar_block_base,
    uint32_t n_blocks_chunk, uint32_t k_stride, uint32_t batch,
    uint32_t blk_tokens, uint32_t first_block_token_offset,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t rope_dim,
    int32_t rope_base, float theta, uint32_t n_subblocks,
    cudaStream_t stream);
bool launch_block_kmean_content_batch_typed(
    const float *k_batch, void *kbar, KbarDType kbar_dtype,
    uint64_t kbar_block_base, uint32_t n_blocks_chunk,
    uint32_t k_stride, uint32_t batch, uint32_t blk_tokens,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t rope_dim,
    int32_t rope_base, float theta, uint32_t n_subblocks,
    cudaStream_t stream);
bool launch_block_kmean_content_batch_merge_typed(
    const float *k_batch, void *kbar, KbarDType kbar_dtype,
    uint64_t kbar_block_base, uint32_t n_blocks_chunk,
    uint32_t k_stride, uint32_t batch, uint32_t blk_tokens,
    uint32_t first_block_token_offset, uint32_t n_kv_heads,
    uint32_t head_dim, uint32_t rope_dim, int32_t rope_base,
    float theta, uint32_t n_subblocks, cudaStream_t stream);
}

#define CHECK(call) do {                                                   \
    cudaError_t err_ = (call);                                             \
    if (err_ != cudaSuccess) {                                             \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(err_));                            \
        std::exit(1);                                                      \
    }                                                                      \
} while (0)

static void require_launch(bool ok, const char *what) {
    if (!ok) {
        std::fprintf(stderr, "launcher rejected %s\n", what);
        std::exit(1);
    }
    CHECK(cudaDeviceSynchronize());
}

static void compare(const std::vector<float> &got,
                    const std::vector<float> &want,
                    const char *label,
                    double limit = 2e-5) {
    double max_abs = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double e = std::fabs(static_cast<double>(got[i]) - want[i]);
        if (e > max_abs) {
            max_abs = e;
            worst = i;
        }
    }
    std::printf("[kvmem-kmean-merge] %-24s max_abs=%.3e\n", label, max_abs);
    if (max_abs > limit) {
        std::fprintf(stderr, "FAIL %s at %zu: got %.9g want %.9g\n",
                     label, worst, got[worst], want[worst]);
        std::exit(1);
    }
}

template <typename T>
__global__ void dequantize_mean(const T *src, float *dst, uint64_t count) {
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
         i < count;
         i += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        dst[i] = static_cast<float>(src[i]);
    }
}

static void run_case(uint32_t subblocks) {
    constexpr uint32_t rows = 19;
    constexpr uint32_t bt = 8;
    constexpr uint32_t heads = 2;
    constexpr uint32_t dim = 8;
    constexpr uint32_t rope_dim = 8;
    constexpr float theta = 10000.0f;
    constexpr uint32_t blocks = (rows + bt - 1) / bt;
    constexpr uint32_t stride = heads * dim;
    const size_t kbar_count = blocks * subblocks * heads * dim;

    std::mt19937 rng(991 + subblocks);
    std::normal_distribution<float> nd(0.0f, 0.4f);
    std::vector<float> input(rows * stride);
    for (float &v : input) v = nd(rng);

    float *d_input = nullptr;
    float *d_ref = nullptr;
    float *d_split = nullptr;
    CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_ref, kbar_count * sizeof(float)));
    CHECK(cudaMalloc(&d_split, kbar_count * sizeof(float)));
    CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_ref, 0, kbar_count * sizeof(float)));
    CHECK(cudaMemset(d_split, 0, kbar_count * sizeof(float)));

    require_launch(qw3::ported::launch_block_kmean_content_batch(
        d_input, d_ref, 0, blocks, stride, rows, bt, heads, dim, rope_dim,
        0, theta, subblocks, 0), "one-shot");

    // [0,3), [3,10), [10,19): both split points are inside logical blocks.
    require_launch(qw3::ported::launch_block_kmean_content_batch(
        d_input, d_split, 0, 1, stride, 3, bt, heads, dim, rope_dim,
        0, theta, subblocks, 0), "split head");
    require_launch(qw3::ported::launch_block_kmean_content_batch_merge(
        d_input + 3 * stride, d_split, 0, 2, stride, 7, bt, 3,
        heads, dim, rope_dim, 3, theta, subblocks, 0), "split middle");
    require_launch(qw3::ported::launch_block_kmean_content_batch_merge(
        d_input + 10 * stride, d_split, 1, 2, stride, 9, bt, 2,
        heads, dim, rope_dim, 10, theta, subblocks, 0), "split tail");

    std::vector<float> ref(kbar_count), split(kbar_count);
    CHECK(cudaMemcpy(ref.data(), d_ref, kbar_count * sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(split.data(), d_split, kbar_count * sizeof(float),
                     cudaMemcpyDeviceToHost));
    compare(split, ref, subblocks == 1 ? "plain split" : "subblock split");

    // Replay [8,19) with different K rows. An aligned write must overwrite the
    // two suffix blocks while leaving block zero untouched.
    std::vector<float> replay = input;
    for (uint32_t r = 8; r < rows; ++r) {
        for (uint32_t i = 0; i < stride; ++i) replay[r * stride + i] = nd(rng);
    }
    CHECK(cudaMemcpy(d_input, replay.data(), replay.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
    require_launch(qw3::ported::launch_block_kmean_content_batch(
        d_input, d_ref, 0, blocks, stride, rows, bt, heads, dim, rope_dim,
        0, theta, subblocks, 0), "replay reference");
    require_launch(qw3::ported::launch_block_kmean_content_batch(
        d_input + 8 * stride, d_split, 1, 2, stride, rows - 8, bt,
        heads, dim, rope_dim, 8, theta, subblocks, 0), "replay overwrite");
    CHECK(cudaMemcpy(ref.data(), d_ref, kbar_count * sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(split.data(), d_split, kbar_count * sizeof(float),
                     cudaMemcpyDeviceToHost));
    compare(split, ref, subblocks == 1 ? "plain replay" : "subblock replay");

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_ref));
    CHECK(cudaFree(d_split));
}

template <typename T>
static void run_typed_case(qw3::ported::KbarDType dtype,
                           const char *dtype_name,
                           uint32_t subblocks,
                           double limit) {
    constexpr uint32_t rows = 19;
    constexpr uint32_t bt = 8;
    constexpr uint32_t heads = 2;
    constexpr uint32_t dim = 8;
    constexpr uint32_t rope_dim = 8;
    constexpr float theta = 10000.0f;
    constexpr uint32_t blocks = (rows + bt - 1) / bt;
    constexpr uint32_t stride = heads * dim;
    const size_t kbar_count = blocks * subblocks * heads * dim;

    std::mt19937 rng(1991 + subblocks + static_cast<uint32_t>(dtype) * 101);
    std::normal_distribution<float> nd(0.0f, 0.4f);
    std::vector<float> input(rows * stride);
    for (float &v : input) v = nd(rng);

    float *d_input = nullptr;
    T *d_ref = nullptr;
    T *d_split = nullptr;
    float *d_ref_f32 = nullptr;
    float *d_split_f32 = nullptr;
    CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_ref, kbar_count * sizeof(T)));
    CHECK(cudaMalloc(&d_split, kbar_count * sizeof(T)));
    CHECK(cudaMalloc(&d_ref_f32, kbar_count * sizeof(float)));
    CHECK(cudaMalloc(&d_split_f32, kbar_count * sizeof(float)));
    CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_ref, 0, kbar_count * sizeof(T)));
    CHECK(cudaMemset(d_split, 0, kbar_count * sizeof(T)));

    require_launch(qw3::ported::launch_block_kmean_content_batch_typed(
        d_input, d_ref, dtype, 0, blocks, stride, rows, bt, heads, dim,
        rope_dim, 0, theta, subblocks, 0), "typed one-shot");
    require_launch(qw3::ported::launch_block_kmean_content_batch_typed(
        d_input, d_split, dtype, 0, 1, stride, 3, bt, heads, dim, rope_dim,
        0, theta, subblocks, 0), "typed split head");
    require_launch(qw3::ported::launch_block_kmean_content_batch_merge_typed(
        d_input + 3 * stride, d_split, dtype, 0, 2, stride, 7, bt, 3,
        heads, dim, rope_dim, 3, theta, subblocks, 0), "typed split middle");
    require_launch(qw3::ported::launch_block_kmean_content_batch_merge_typed(
        d_input + 10 * stride, d_split, dtype, 1, 2, stride, 9, bt, 2,
        heads, dim, rope_dim, 10, theta, subblocks, 0), "typed split tail");

    const uint32_t threads = 128;
    const uint32_t grid =
        static_cast<uint32_t>((kbar_count + threads - 1) / threads);
    dequantize_mean<<<grid, threads>>>(d_ref, d_ref_f32, kbar_count);
    dequantize_mean<<<grid, threads>>>(d_split, d_split_f32, kbar_count);
    CHECK(cudaDeviceSynchronize());
    std::vector<float> ref(kbar_count), split(kbar_count);
    CHECK(cudaMemcpy(ref.data(), d_ref_f32, kbar_count * sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(split.data(), d_split_f32, kbar_count * sizeof(float),
                     cudaMemcpyDeviceToHost));
    char label[80];
    std::snprintf(label, sizeof(label), "%s split ns=%u",
                  dtype_name, subblocks);
    compare(split, ref, label, limit);

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_ref));
    CHECK(cudaFree(d_split));
    CHECK(cudaFree(d_ref_f32));
    CHECK(cudaFree(d_split_f32));
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::fprintf(stderr, "[kvmem-kmean-merge] no CUDA device; skipping\n");
        return 0;
    }
    run_case(1);
    run_case(4);
    run_typed_case<__half>(
        qw3::ported::KbarDType::F16, "fp16", 1, 1e-3);
    run_typed_case<__half>(
        qw3::ported::KbarDType::F16, "fp16", 4, 1e-3);
    run_typed_case<__nv_fp8_e4m3>(
        qw3::ported::KbarDType::FP8, "fp8", 1, 7e-2);
    run_typed_case<__nv_fp8_e4m3>(
        qw3::ported::KbarDType::FP8, "fp8", 4, 7e-2);
    std::puts("[kvmem-kmean-merge] PASS");
    return 0;
}
