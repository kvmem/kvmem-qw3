// MMQ v4 parity test.
//
// Validates that the v4 kernel (qw3::ported::launch_mmq_q8_0 with
// QW3_MMQ_VERSION=4) produces the same matmul result that an FP32 host
// reference computes from the same dequantized Q8_0 weights and the same
// dequantized Q8_1_MMQ activations the kernel sees.
//
// Strategy:
//   1. Generate random FP32 weights W[M,K] and activations X[N,K].
//   2. Quantize W into Q8_0 on the host (34 B/block).
//   3. Stage W on device, run launch_quantize_mmq_q8_1 to produce Q8_1_MMQ
//      activations (144 B/block, super-block-major × j-minor) on device.
//   4. Read both buffers back, dequant on host to W_de[M,K] and X_de[N,K],
//      and compute dst_ref[n,m] = sum_k W_de[m,k] * X_de[n,k] in FP32.
//   5. Run launch_mmq_q8_0 on device, copy dst back, compare to dst_ref.
//
// This isolates v4 matmul correctness from the quantizers, both of which
// have separate parity coverage in the v2/v3 path / mmvq tests.

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

namespace qw3 { namespace ported {
size_t q8_1_mmq_scratch_bytes(uint32_t batch, uint32_t cols);
bool launch_quantize_mmq_q8_1(const float *x, void *y_q8_1_mmq,
                              uint32_t batch, uint32_t cols, uint32_t stride_x_row,
                              cudaStream_t stream);
bool launch_mmq_q8_0(const uint8_t *weight, const void *y_q8_1, float *dst,
                     uint32_t rows, uint32_t cols, uint32_t batch,
                     uint32_t stride_dst_row, cudaStream_t stream);
}}

#define CHECK(call) do {                                                  \
    cudaError_t _err = (call);                                            \
    if (_err != cudaSuccess) {                                            \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",                    \
                     __FILE__, __LINE__, cudaGetErrorString(_err));       \
        std::exit(1);                                                     \
    }                                                                     \
} while (0)

static constexpr int QK = 32;            // Q8_0 / Q8_1 block size
static constexpr int MMQ_BLK = 128;      // Q8_1_MMQ super-block size

static void quantize_q8_0_host(const std::vector<float> &W,
                               int M, int K,
                               std::vector<uint8_t> &out_blocks) {
    const int n_blocks_per_row = K / QK;
    out_blocks.assign(static_cast<size_t>(M) * n_blocks_per_row * 34, 0u);
    for (int m = 0; m < M; ++m) {
        for (int kb = 0; kb < n_blocks_per_row; ++kb) {
            float amax = 0.0f;
            for (int t = 0; t < QK; ++t) {
                amax = std::max(amax, std::fabs(W[m * K + kb * QK + t]));
            }
            const float d = amax / 127.0f;
            const float d_inv = (d == 0.0f) ? 0.0f : 1.0f / d;
            uint8_t *blk = out_blocks.data() +
                (static_cast<size_t>(m) * n_blocks_per_row + kb) * 34;
            __half hd = __float2half(d);
            std::memcpy(blk, &hd, 2);
            int8_t *qs = reinterpret_cast<int8_t *>(blk + 2);
            for (int t = 0; t < QK; ++t) {
                float q = W[m * K + kb * QK + t] * d_inv;
                int qi = static_cast<int>(std::lrintf(q));
                qi = std::max(-127, std::min(127, qi));
                qs[t] = static_cast<int8_t>(qi);
            }
        }
    }
}

static void dequant_q8_0_host(const std::vector<uint8_t> &Wq,
                              int M, int K,
                              std::vector<float> &out) {
    const int n_blocks_per_row = K / QK;
    out.assign(static_cast<size_t>(M) * K, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int kb = 0; kb < n_blocks_per_row; ++kb) {
            const uint8_t *blk = Wq.data() +
                (static_cast<size_t>(m) * n_blocks_per_row + kb) * 34;
            __half hd; std::memcpy(&hd, blk, 2);
            float d = __half2float(hd);
            const int8_t *qs = reinterpret_cast<const int8_t *>(blk + 2);
            for (int t = 0; t < QK; ++t) {
                out[m * K + kb * QK + t] = qs[t] * d;
            }
        }
    }
}

// Layout per src/mmvq_q8.cu and src/mmq_q8.cu:
//   block_q8_1_mmq_t = { float d4[4]; int8_t qs[128]; }   // 144 bytes
//   ib = super_block * ncols_y + j
//   qs[s*32 + t] dequants with d4[s]
//   K position covered by block ib at sub-index (s*32 + t) is
//     k_global = super_block*128 + s*32 + t
static void dequant_q8_1_mmq_host(const std::vector<uint8_t> &Yq,
                                  int N, int K,
                                  std::vector<float> &out) {
    const int sblocks_per_row = K / MMQ_BLK;
    out.assign(static_cast<size_t>(N) * K, 0.0f);
    for (int j = 0; j < N; ++j) {
        for (int sb = 0; sb < sblocks_per_row; ++sb) {
            const size_t ib = static_cast<size_t>(sb) * N + j;
            const uint8_t *blk = Yq.data() + ib * 144;
            float d4[4];
            std::memcpy(d4, blk, 16);
            const int8_t *qs = reinterpret_cast<const int8_t *>(blk + 16);
            for (int s = 0; s < 4; ++s) {
                for (int t = 0; t < QK; ++t) {
                    out[j * K + sb * MMQ_BLK + s * QK + t] = qs[s * QK + t] * d4[s];
                }
            }
        }
    }
}

struct ShapeResult {
    int M, N, K;
    float max_abs_diff;
    float max_rel_diff;
    int n_outside_tol;
    int total;
};

static ShapeResult run_shape(int M, int N, int K, std::mt19937 &rng) {
    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dw(-0.5f, 0.5f);

    std::vector<float> W(static_cast<size_t>(M) * K);
    std::vector<float> X(static_cast<size_t>(N) * K);
    for (auto &v : W) v = dw(rng);
    for (auto &v : X) v = dx(rng);

    std::vector<uint8_t> Wq;
    quantize_q8_0_host(W, M, K, Wq);

    // Repack Wq into the split-plane layout the device kernels now expect:
    //   per row: [FP16 scales × n_blocks][int8 quants × n_blocks * 32]
    // Same total bytes per row; matches CudaWeight's Q8_0 ctor in src/kernels_cuda.cu.
    std::vector<uint8_t> Wq_split(Wq.size(), 0u);
    {
        const int n_blocks_per_row = K / QK;
        const size_t row_bytes = static_cast<size_t>(n_blocks_per_row) * 34;
        for (int m = 0; m < M; ++m) {
            const uint8_t *in_row  = Wq.data() + static_cast<size_t>(m) * row_bytes;
            uint8_t       *out_row = Wq_split.data() + static_cast<size_t>(m) * row_bytes;
            uint8_t *plane_d = out_row;
            uint8_t *plane_q = out_row + 2 * static_cast<size_t>(n_blocks_per_row);
            for (int b = 0; b < n_blocks_per_row; ++b) {
                const uint8_t *blk = in_row + static_cast<size_t>(b) * 34;
                plane_d[2 * b + 0] = blk[0];
                plane_d[2 * b + 1] = blk[1];
                std::memcpy(plane_q + static_cast<size_t>(b) * 32, blk + 2, 32);
            }
        }
    }

    uint8_t *d_w = nullptr;
    float   *d_x = nullptr;
    float   *d_out = nullptr;
    void    *d_act = nullptr;

    CHECK(cudaMalloc(&d_w, Wq_split.size()));
    CHECK(cudaMemcpy(d_w, Wq_split.data(), Wq_split.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_x, X.size() * sizeof(float)));
    CHECK(cudaMemcpy(d_x, X.data(), X.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_out, static_cast<size_t>(N) * M * sizeof(float)));
    CHECK(cudaMemset(d_out, 0, static_cast<size_t>(N) * M * sizeof(float)));

    const size_t scratch_bytes =
        qw3::ported::q8_1_mmq_scratch_bytes(static_cast<uint32_t>(N),
                                            static_cast<uint32_t>(K));
    CHECK(cudaMalloc(&d_act, scratch_bytes));

    if (!qw3::ported::launch_quantize_mmq_q8_1(
                d_x, d_act,
                static_cast<uint32_t>(N), static_cast<uint32_t>(K),
                static_cast<uint32_t>(K), /*stream=*/0)) {
        std::fprintf(stderr, "launch_quantize_mmq_q8_1 failed (M=%d N=%d K=%d)\n",
                     M, N, K);
        std::exit(1);
    }

    if (!qw3::ported::launch_mmq_q8_0(
                d_w, d_act, d_out,
                static_cast<uint32_t>(M), static_cast<uint32_t>(K),
                static_cast<uint32_t>(N),
                /*stride_dst_row=*/static_cast<uint32_t>(M),
                /*stream=*/0)) {
        std::fprintf(stderr, "launch_mmq_q8_0 failed (M=%d N=%d K=%d)\n",
                     M, N, K);
        std::exit(1);
    }

    CHECK(cudaDeviceSynchronize());
    cudaError_t kerr = cudaGetLastError();
    if (kerr != cudaSuccess) {
        std::fprintf(stderr, "kernel error %s (M=%d N=%d K=%d)\n",
                     cudaGetErrorString(kerr), M, N, K);
        std::exit(1);
    }

    std::vector<uint8_t> Yq(scratch_bytes);
    CHECK(cudaMemcpy(Yq.data(), d_act, scratch_bytes, cudaMemcpyDeviceToHost));

    std::vector<float> got(static_cast<size_t>(N) * M);
    CHECK(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));

    cudaFree(d_w); cudaFree(d_x); cudaFree(d_out); cudaFree(d_act);

    // Reference using kernel-seen (dequantized) operands.
    std::vector<float> W_de;
    std::vector<float> X_de;
    dequant_q8_0_host(Wq, M, K, W_de);
    dequant_q8_1_mmq_host(Yq, N, K, X_de);

    std::vector<float> ref(static_cast<size_t>(N) * M, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            float s = 0.0f;
            const float *wrow = W_de.data() + static_cast<size_t>(m) * K;
            const float *xrow = X_de.data() + static_cast<size_t>(n) * K;
            for (int k = 0; k < K; ++k) s += wrow[k] * xrow[k];
            ref[static_cast<size_t>(n) * M + m] = s;
        }
    }

    ShapeResult r{M, N, K, 0.0f, 0.0f, 0, static_cast<int>(got.size())};
    // Tolerance: kernel accumulates in S32 and rescales by per-block FP32
    // scales after every K subblock — order-of-summation differs slightly
    // from the host loop. 5e-3 abs / 1e-3 rel is comfortably inside that drift
    // for K up to a few thousand at our magnitudes.
    const float abs_tol = 5e-3f;
    const float rel_tol = 1e-3f;
    for (size_t i = 0; i < got.size(); ++i) {
        float d = std::fabs(got[i] - ref[i]);
        r.max_abs_diff = std::max(r.max_abs_diff, d);
        float denom = std::max(std::fabs(ref[i]), 1e-6f);
        r.max_rel_diff = std::max(r.max_rel_diff, d / denom);
        if (d > abs_tol && d > rel_tol * denom) ++r.n_outside_tol;
    }
    return r;
}

int main(int /*argc*/, char ** /*argv*/) {
    // Force MMQ v7 dispatch in launch_mmq_q8_0 (consumes 144-B block_q8_1_mmq_t).
    setenv("QW3_MMQ_VERSION", "7", 1);

    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::fprintf(stderr, "no CUDA devices, skipping\n");
        return 0;
    }

    std::mt19937 rng(0x5eedu);

    struct Shape { int M, N, K; };
    // v4 dispatch requires rows>=128, batch>=128, cols%256==0.
    std::vector<Shape> shapes = {
        {128, 128,  256},
        {128, 128, 1024},
        {128, 128, 2048},
        {256, 128, 2048},
        {128, 256, 2048},
        {256, 256, 1024},
        {384, 256, 2048},
    };

    int n_failed = 0;
    for (auto s : shapes) {
        ShapeResult r = run_shape(s.M, s.N, s.K, rng);
        std::printf("M=%4d N=%4d K=%5d  max_abs=%.4g  max_rel=%.4g  outside_tol=%d/%d\n",
                    r.M, r.N, r.K, r.max_abs_diff, r.max_rel_diff,
                    r.n_outside_tol, r.total);
        if (r.n_outside_tol != 0) ++n_failed;
    }

    if (n_failed != 0) {
        std::printf("FAILED: %d shape(s) outside tolerance\n", n_failed);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
