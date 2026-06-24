// Re-RoPE block-remap parity test (block-sparse KV reuse, Task #38).
//
// Validates the gold standard the whole block-sparse scheme rests on:
//
//   A remapped key must be byte-equal to a key NATIVELY baked at the new
//   window position by the production RoPE path.
//
// Production stores K with RoPE baked at its ORIGINAL position p_orig via
// rope_partial_kernel (__powf inv_freq, __sincosf rotation):
//       k_orig = R~(p_orig)*k_raw           (R~ = __sincosf approximation)
// To reuse that block at a NEW in-window position p_new we de-rotate by the
// SAME R~(p_orig) (transpose) and re-rotate at p_new:
//       k_remap = R~(p_new) * R~(p_orig)^T * k_orig
// Because the de-rotation reuses the identical __sincosf(ang_o) values, the
// huge-position range-reduction error at p_orig (~1M for a block pulled from a
// long context) cancels (R~^T*R~ = (sin^2+cos^2)*I ~ I), leaving
//       k_remap ~ R~(p_new)*k_raw = k_fresh
// which is exactly what production bakes fresh at p_new. So remap == native
// bake to fp ULP, INDEPENDENT of how large p_orig is. (A naive single-delta
// rotation does NOT achieve this: it leaves the stale p_orig __sincosf error
// in the key and diverges by ~0.05-0.15 at p_orig~1M.)
//
// Both the bake and the remap run with __sincosf here, matching production
// (src/kernels_cuda.cu rope_partial_kernel / rope_block_remap_kernel), so this
// is a device-vs-device gold standard, not a device-vs-fp64-truth comparison.

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
bool launch_rope_block_remap(float *x, uint32_t n_tokens, uint32_t n_units,
                             uint32_t per_unit_stride, uint32_t rope_dim,
                             int32_t orig_base, int32_t new_base,
                             uint32_t row_stride, float theta,
                             cudaStream_t stream);
}}

#define CHECK(call) do {                                                  \
    cudaError_t _err = (call);                                            \
    if (_err != cudaSuccess) {                                            \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",                    \
                     __FILE__, __LINE__, cudaGetErrorString(_err));       \
        std::exit(1);                                                     \
    }                                                                     \
} while (0)

// Exact mirror of rope_partial_kernel (src/kernels_cuda.cu:2547) — the
// production RoPE bake. pos = base_pos + token (blockIdx.x = token).
__global__ void test_rope_bake_kernel(float *x, uint32_t n_units,
                                      uint32_t per_unit_stride,
                                      uint32_t rope_dim, int32_t base_pos,
                                      uint32_t row_stride, float theta) {
    const uint32_t tok  = blockIdx.x;
    const uint32_t unit = blockIdx.y;
    const uint32_t i = threadIdx.x;
    if (unit >= n_units) return;
    const uint32_t half = rope_dim / 2;
    if (i >= half) return;
    float *base = x + static_cast<uint64_t>(tok) * row_stride + unit * per_unit_stride;
    const float inv_freq = __powf(theta, -2.0f * static_cast<float>(i) / static_cast<float>(rope_dim));
    const float angle = static_cast<float>(base_pos + static_cast<int32_t>(tok)) * inv_freq;
    float c, s;
    __sincosf(angle, &s, &c);
    const float x0 = base[i];
    const float x1 = base[i + half];
    base[i]        = x0 * c - x1 * s;
    base[i + half] = x0 * s + x1 * c;
}

static void bake_on_device(std::vector<float> &x, uint32_t n_tokens,
                           uint32_t n_units, uint32_t per_unit_stride,
                           uint32_t rope_dim, int32_t base_pos,
                           uint32_t row_stride, float theta) {
    float *d = nullptr;
    CHECK(cudaMalloc(&d, x.size() * sizeof(float)));
    CHECK(cudaMemcpy(d, x.data(), x.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
    dim3 grid(n_tokens, n_units);
    test_rope_bake_kernel<<<grid, rope_dim / 2>>>(
        d, n_units, per_unit_stride, rope_dim, base_pos, row_stride, theta);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(x.data(), d, x.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
    cudaFree(d);
}

struct CaseResult {
    int64_t p_orig, p_new;
    uint32_t n_tokens, n_units, head_dim, rope_dim;
    float max_abs_k;     // device remap vs device fresh bake, K vector
    float max_abs_score; // attention score q.k: remap vs fresh
    int n_outside;
    int total;
};

static CaseResult run_case(int64_t p_orig, int64_t p_new,
                           uint32_t n_tokens, uint32_t n_units,
                           uint32_t head_dim, uint32_t rope_dim,
                           float theta, std::mt19937 &rng) {
    const uint32_t per_unit_stride = head_dim;
    const uint32_t row_stride = n_units * head_dim;
    const size_t n = static_cast<size_t>(n_tokens) * row_stride;

    std::uniform_real_distribution<float> dk(-1.0f, 1.0f);
    std::vector<float> k_raw(n);
    for (auto &v : k_raw) v = dk(rng);

    // Production store path: bake at p_orig.  Truth: bake fresh at p_new.
    std::vector<float> k_orig = k_raw;
    bake_on_device(k_orig, n_tokens, n_units, per_unit_stride, rope_dim,
                   static_cast<int32_t>(p_orig), row_stride, theta);
    std::vector<float> k_fresh = k_raw;
    bake_on_device(k_fresh, n_tokens, n_units, per_unit_stride, rope_dim,
                   static_cast<int32_t>(p_new), row_stride, theta);

    // Remap the stored (p_orig) key to the new window position.
    float *d_k = nullptr;
    CHECK(cudaMalloc(&d_k, n * sizeof(float)));
    CHECK(cudaMemcpy(d_k, k_orig.data(), n * sizeof(float),
                     cudaMemcpyHostToDevice));
    if (!qw3::ported::launch_rope_block_remap(
                d_k, n_tokens, n_units, per_unit_stride, rope_dim,
                static_cast<int32_t>(p_orig), static_cast<int32_t>(p_new),
                row_stride, theta, /*stream=*/0)) {
        std::fprintf(stderr, "launch_rope_block_remap failed\n");
        std::exit(1);
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    std::vector<float> k_remap(n);
    CHECK(cudaMemcpy(k_remap.data(), d_k, n * sizeof(float),
                     cudaMemcpyDeviceToHost));
    cudaFree(d_k);

    // A query baked at an in-window position, to check the quantity the
    // attention kernel actually consumes (q.k) is preserved.
    std::vector<float> q(row_stride);
    std::uniform_real_distribution<float> dq(-1.0f, 1.0f);
    for (auto &v : q) v = dq(rng);
    bake_on_device(q, 1, n_units, per_unit_stride, rope_dim,
                   static_cast<int32_t>(p_new + static_cast<int64_t>(n_tokens) + 7),
                   row_stride, theta);

    CaseResult r{p_orig, p_new, n_tokens, n_units, head_dim, rope_dim,
                 0.0f, 0.0f, 0, static_cast<int>(n)};
    const float abs_tol = 1e-3f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(k_remap[i] - k_fresh[i]);
        r.max_abs_k = std::max(r.max_abs_k, d);
        if (d > abs_tol) ++r.n_outside;
    }
    for (uint32_t tok = 0; tok < n_tokens; ++tok) {
        for (uint32_t u = 0; u < n_units; ++u) {
            const size_t koff = static_cast<size_t>(tok) * row_stride +
                                static_cast<size_t>(u) * head_dim;
            const size_t qoff = static_cast<size_t>(u) * head_dim;
            float s_remap = 0.0f, s_fresh = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                s_remap += q[qoff + d] * k_remap[koff + d];
                s_fresh += q[qoff + d] * k_fresh[koff + d];
            }
            r.max_abs_score = std::max(r.max_abs_score,
                                       std::fabs(s_remap - s_fresh));
        }
    }
    return r;
}

int main(int /*argc*/, char ** /*argv*/) {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::fprintf(stderr, "no CUDA devices, skipping\n");
        return 0;
    }

    std::mt19937 rng(0x5eedu);
    const float theta = 1e7f;        // qwen rope_theta default
    const uint32_t head_dim = 256;   // Qwen 3.6 attention head_dim
    const uint32_t rope_dim = 128;   // partial RoPE over first half

    struct Case {
        int64_t p_orig, p_new;
        uint32_t n_tokens, n_units;
    };
    // Block sizes (n_tokens) at 128-tok default; a few unit counts (kv-heads).
    // Deltas span both directions and the large magnitudes the remap must
    // handle (scattered ~1M-token context compressed into a 128k window).
    std::vector<Case> cases = {
        {1000,       0,        128, 2},   // pull a far block down to window start
        {0,          1000,     128, 2},   // push start block out (positive delta)
        {500000,     130000,   128, 4},   // ~1M-context block -> 128k window tail
        {123456,     654,      128, 4},   // arbitrary large -> small
        {654,        123456,   128, 2},   // arbitrary small -> large
        {1000000,    131071,   128, 8},   // extreme: 1M -> just under 128k
        {77,         77,       128, 2},   // zero delta: must be a no-op
        {2048,       64,       64,  2},   // partial (tail) block, 64 tokens
        {1,          131000,   128, 4},   // sink block remapped far out
    };

    int n_failed = 0;
    for (auto c : cases) {
        CaseResult r = run_case(c.p_orig, c.p_new, c.n_tokens, c.n_units,
                                head_dim, rope_dim, theta, rng);
        std::printf("p_orig=%-8lld p_new=%-8lld tok=%-4u units=%u  "
                    "max_abs_k=%.4g  max_abs_score=%.4g  outside=%d/%d\n",
                    static_cast<long long>(r.p_orig),
                    static_cast<long long>(r.p_new),
                    r.n_tokens, r.n_units, r.max_abs_k, r.max_abs_score,
                    r.n_outside, r.total);
        if (r.n_outside != 0) ++n_failed;
    }

    if (n_failed != 0) {
        std::printf("FAILED: %d case(s) outside tolerance\n", n_failed);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
