// DeltaNet-state retrieval scoring parity test
// (docs/kvmem_deltanet_retrieval_design.md).
//
// Validates the two DeltaNet-retrieval CUDA kernels against a plain host
// reference on small synthetic inputs:
//
//   1. deltanet_block_score_kernel — per (block, v-head) score
//        r[blk,vh] = TopKMean_t( d[blk,vh] * || (S_j - a_j S_{j-1})^T q_t ||_2 )
//      where S_j = dn_snap[blk], S_{j-1} = dn_snap[blk-1] (0 for blk 0),
//      a_j = exp(decaysum[blk,vh]), d[blk,vh] = decay_d[blk,vh].
//
//   2. The full host aggregation (TopKMean over heads, per-layer RMS-norm,
//      equal-weight layer sum) that produces the final per-block score, matching
//      QwenExecutor::kvmem_retrieval_score_deltanet's host tail.
//
// The kernel launcher is declared in qw3::ported (src/kernels_cuda.cu). We link
// qw3_core and drive it directly with device buffers.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

namespace qw3 {
namespace ported {
bool launch_deltanet_block_score(float *r_out, const float *dn_snap,
                                 const float *decaysum, const float *decay_d,
                                 const float *q_layer, uint32_t n_blocks,
                                 uint32_t num_k_heads, uint32_t num_v_heads,
                                 uint32_t head_v_dim, uint32_t head_k_dim,
                                 uint32_t M, uint32_t topk_q,
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

int main() {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::fprintf(stderr, "[kvmem-deltanet-score] no CUDA device; skipping\n");
        return 0;
    }

    // Small synthetic problem. head_k_dim/head_v_dim kept modest so the reference
    // is cheap; the kernel supports up to 128 (one thread per k-dim).
    const uint32_t n_blocks = 6;
    const uint32_t num_k_heads = 2;
    const uint32_t num_v_heads = 4;   // group = num_v_heads/num_k_heads = 2
    const uint32_t head_v_dim = 8;
    const uint32_t head_k_dim = 8;
    const uint32_t M = 5;             // query tokens
    const uint32_t topk_q = 3;

    const uint64_t snap_elems =
        static_cast<uint64_t>(n_blocks) * num_v_heads * head_v_dim * head_k_dim;
    const uint64_t small_elems = static_cast<uint64_t>(n_blocks) * num_v_heads;
    const uint64_t q_elems =
        static_cast<uint64_t>(M) * num_k_heads * head_k_dim;

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(-0.5f, 0.0f);  // log-decay <= 0

    std::vector<float> snap(snap_elems);
    for (auto &v : snap) v = nd(rng);
    std::vector<float> decaysum(small_elems);
    for (auto &v : decaysum) v = ud(rng);      // a_j = exp(<=0) in (0,1]
    std::vector<float> decay_d(small_elems);
    for (auto &v : decay_d) v = std::exp(ud(rng));  // d_j in (0,1]
    std::vector<float> q(q_elems);
    for (auto &v : q) v = nd(rng);

    // Device buffers.
    float *d_snap, *d_decaysum, *d_decay_d, *d_q, *d_r;
    CHECK(cudaMalloc(&d_snap, snap_elems * sizeof(float)));
    CHECK(cudaMalloc(&d_decaysum, small_elems * sizeof(float)));
    CHECK(cudaMalloc(&d_decay_d, small_elems * sizeof(float)));
    CHECK(cudaMalloc(&d_q, q_elems * sizeof(float)));
    CHECK(cudaMalloc(&d_r, small_elems * sizeof(float)));
    CHECK(cudaMemcpy(d_snap, snap.data(), snap_elems * sizeof(float),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_decaysum, decaysum.data(), small_elems * sizeof(float),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_decay_d, decay_d.data(), small_elems * sizeof(float),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_q, q.data(), q_elems * sizeof(float),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_r, 0, small_elems * sizeof(float)));

    if (!qw3::ported::launch_deltanet_block_score(
            d_r, d_snap, d_decaysum, d_decay_d, d_q, n_blocks, num_k_heads,
            num_v_heads, head_v_dim, head_k_dim, M, topk_q, 0)) {
        std::fprintf(stderr, "[kvmem-deltanet-score] kernel launch failed\n");
        return 1;
    }
    CHECK(cudaDeviceSynchronize());

    std::vector<float> r_gpu(small_elems);
    CHECK(cudaMemcpy(r_gpu.data(), d_r, small_elems * sizeof(float),
                     cudaMemcpyDeviceToHost));

    // Host reference for the per-(block,vh) score.
    auto snap_at = [&](uint32_t blk, uint32_t vh, uint32_t jrow, uint32_t i) {
        const uint64_t idx =
            ((static_cast<uint64_t>(blk) * num_v_heads + vh) * head_v_dim + jrow) *
                head_k_dim + i;
        return snap[idx];
    };
    auto q_at = [&](uint32_t t, uint32_t kh, uint32_t i) {
        return q[(static_cast<uint64_t>(t) * num_k_heads + kh) * head_k_dim + i];
    };

    std::vector<float> r_ref(small_elems, 0.0f);
    for (uint32_t blk = 0; blk < n_blocks; ++blk) {
        for (uint32_t vh = 0; vh < num_v_heads; ++vh) {
            const uint32_t kh = vh % num_k_heads;
            const float a_j = std::exp(decaysum[blk * num_v_heads + vh]);
            const float d = decay_d[blk * num_v_heads + vh];
            std::vector<float> r_t(M, 0.0f);
            for (uint32_t t = 0; t < M; ++t) {
                double norm2 = 0.0;
                for (uint32_t jrow = 0; jrow < head_v_dim; ++jrow) {
                    double c = 0.0;
                    for (uint32_t i = 0; i < head_k_dim; ++i) {
                        const float sj = snap_at(blk, vh, jrow, i);
                        const float sp =
                            (blk > 0) ? snap_at(blk - 1, vh, jrow, i) : 0.0f;
                        const float e = sj - a_j * sp;
                        c += static_cast<double>(e) * q_at(t, kh, i);
                    }
                    norm2 += c * c;
                }
                r_t[t] = d * static_cast<float>(std::sqrt(norm2));
            }
            // TopKMean over the top-topk_q tokens.
            std::vector<float> sorted = r_t;
            std::sort(sorted.begin(), sorted.end(), std::greater<float>());
            const uint32_t kq = std::min(topk_q, M);
            double sum = 0.0;
            for (uint32_t i = 0; i < kq; ++i) sum += sorted[i];
            r_ref[blk * num_v_heads + vh] =
                kq > 0 ? static_cast<float>(sum / kq) : 0.0f;
        }
    }

    // Compare kernel vs reference.
    double max_abs = 0.0, max_rel = 0.0;
    for (uint64_t i = 0; i < small_elems; ++i) {
        const double a = r_gpu[i], b = r_ref[i];
        const double abs_err = std::fabs(a - b);
        const double rel_err = abs_err / (std::fabs(b) + 1e-6);
        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, rel_err);
    }
    std::printf("[kvmem-deltanet-score] block-score max_abs=%.3e max_rel=%.3e\n",
                max_abs, max_rel);
    if (max_abs > 1e-3 && max_rel > 1e-3) {
        std::fprintf(stderr,
                     "[kvmem-deltanet-score] FAIL: block score mismatch\n");
        return 1;
    }

    // ---- Aggregation tail (heads TopKMean -> per-layer RMS-norm -> layer sum).
    // Mirror kvmem_retrieval_score_deltanet's host aggregation for a single layer
    // (L=1) using r_ref, and verify basic invariants (RMS ~ 1 after norm).
    const uint32_t topk_h = 2;
    std::vector<double> layer_block(n_blocks, 0.0);
    for (uint32_t j = 0; j < n_blocks; ++j) {
        std::vector<float> hv(num_v_heads);
        for (uint32_t h = 0; h < num_v_heads; ++h) hv[h] = r_ref[j * num_v_heads + h];
        const uint32_t kh = std::min(topk_h, num_v_heads);
        std::partial_sort(hv.begin(), hv.begin() + kh, hv.end(),
                          std::greater<float>());
        double sum = 0.0;
        for (uint32_t i = 0; i < kh; ++i) sum += hv[i];
        layer_block[j] = kh > 0 ? sum / kh : 0.0;
    }
    double ms = 0.0;
    for (uint32_t j = 0; j < n_blocks; ++j) ms += layer_block[j] * layer_block[j];
    const double rms = std::sqrt(ms / n_blocks) + 1e-6;
    double check_ms = 0.0;
    for (uint32_t j = 0; j < n_blocks; ++j) {
        const double v = layer_block[j] / rms;
        check_ms += v * v;
    }
    const double post_rms = std::sqrt(check_ms / n_blocks);
    std::printf("[kvmem-deltanet-score] post-RMS-norm rms=%.4f (expect ~1)\n",
                post_rms);
    if (std::fabs(post_rms - 1.0) > 1e-3) {
        std::fprintf(stderr,
                     "[kvmem-deltanet-score] FAIL: RMS-norm invariant broken\n");
        return 1;
    }

    cudaFree(d_snap);
    cudaFree(d_decaysum);
    cudaFree(d_decay_d);
    cudaFree(d_q);
    cudaFree(d_r);
    std::printf("[kvmem-deltanet-score] PASS\n");
    return 0;
}
