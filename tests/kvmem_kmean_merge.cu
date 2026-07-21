// Incremental mean-K parity test.
//
// Verifies that a logical block built across arbitrary prefill boundaries has
// the same content-frame mean as one-shot capture, including sub-block mode and
// a block-aligned replay overwrite.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace qw3::ported {
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
                    const char *label) {
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
    if (max_abs > 2e-5) {
        std::fprintf(stderr, "FAIL %s at %zu: got %.9g want %.9g\n",
                     label, worst, got[worst], want[worst]);
        std::exit(1);
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

int main() {
    run_case(1);
    run_case(4);
    std::puts("[kvmem-kmean-merge] PASS");
    return 0;
}
