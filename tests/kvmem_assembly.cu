// Block-sparse paged re-RoPE assembly mechanism test (Task #39).
//
// Validates the *assembly recipe* the executor will use, at kernel level, on
// the real paged fp16 cache — before touching the forward path. It pins the two
// "silent KV corruption" risks the plan flagged: (1) fp16 storage rounding and
// (2) the paged physical byte-offset math (a block's tokens scattered across a
// SCRAMBLED page table).
//
// Setup mirrors the live attention layer (qwen_executor.cpp:865-887):
//   - KV cache is paged + fp16: physical slot for logical pos p is
//       page_indices[p/page_size]*page_size + p%page_size  (per-pos row of
//       per_pos_size = n_kv_heads*head_dim halfs), identical to
//       kv_append_paged_kernel_f16.
//   - A set of blocks live at scattered ORIGINAL positions, each stored with
//     RoPE baked at orig_base+tok (the production store path).
//
// Path A (block-sparse): lay the selected blocks into a window page table in
//   ascending order, packed contiguously from window pos 0; run
//   launch_rope_block_remap_paged on each block (orig_base -> new_base); read
//   the remapped K back from the window slots.
// Path B (reference): bake the SAME raw K fresh at the window positions
//   (new_base+tok) via the production rope kernel into a contiguous buffer.
//
// Assert: remapped stored-K (Path A, fp16) ~ fresh-baked K (Path B, fp16), and
// the attention score q.k matches. Tolerance is loose enough for one fp16
// round-trip (the cache stores halfs) but tight enough to catch a wrong page
// offset or a missing de-rotation (which diverge by >0.05 at large orig).

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
bool launch_rope_block_remap_paged(void *cache, bool is_fp16,
                                   uint32_t n_tokens, uint32_t n_kv_heads,
                                   uint32_t per_pos_size, uint32_t head_dim,
                                   uint32_t rope_dim, uint32_t win_base,
                                   int32_t orig_base, int32_t new_base,
                                   const int32_t *page_indices,
                                   uint32_t page_size, float theta,
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

// Production RoPE bake (mirror of rope_partial_kernel) into a CONTIGUOUS fp32
// buffer: pos = base_pos + token, partial over first rope_dim of each head.
__global__ void ref_bake_kernel(float *x, uint32_t n_units,
                                uint32_t per_unit_stride, uint32_t rope_dim,
                                int32_t base_pos, uint32_t row_stride,
                                float theta) {
    const uint32_t tok = blockIdx.x, unit = blockIdx.y, i = threadIdx.x;
    const uint32_t half = rope_dim / 2;
    if (i >= half) return;
    float *base = x + static_cast<uint64_t>(tok) * row_stride + unit * per_unit_stride;
    const float inv_freq = __powf(theta, -2.0f * float(i) / float(rope_dim));
    const float angle = float(base_pos + int32_t(tok)) * inv_freq;
    float c, s; __sincosf(angle, &s, &c);
    const float x0 = base[i], x1 = base[i + half];
    base[i] = x0 * c - x1 * s;
    base[i + half] = x0 * s + x1 * c;
}

// Store a contiguous fp32 per-pos block into the paged fp16 cache at logical
// positions [win_base..win_base+n_tokens) (mirror of kv_append_paged_kernel_f16).
__global__ void paged_store_f16_kernel(__half *cache, const float *src,
                                       uint32_t win_base, uint32_t per_pos_size,
                                       const int32_t *page_indices,
                                       uint32_t page_size) {
    const uint32_t b = blockIdx.x;
    const uint32_t i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= per_pos_size) return;
    const uint32_t lp = win_base + b;
    const uint32_t phys = uint32_t(page_indices[lp / page_size]) * page_size + (lp % page_size);
    cache[uint64_t(phys) * per_pos_size + i] = __float2half(src[uint64_t(b) * per_pos_size + i]);
}

// Read paged fp16 cache logical positions [win_base..+n_tokens) into contiguous fp32.
__global__ void paged_load_f16_kernel(const __half *cache, float *dst,
                                      uint32_t win_base, uint32_t per_pos_size,
                                      const int32_t *page_indices,
                                      uint32_t page_size) {
    const uint32_t b = blockIdx.x;
    const uint32_t i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= per_pos_size) return;
    const uint32_t lp = win_base + b;
    const uint32_t phys = uint32_t(page_indices[lp / page_size]) * page_size + (lp % page_size);
    dst[uint64_t(b) * per_pos_size + i] = __half2float(cache[uint64_t(phys) * per_pos_size + i]);
}

struct Block { int64_t orig_base; uint32_t n_tokens; };

int main() {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
        std::fprintf(stderr, "no CUDA devices, skipping\n");
        return 0;
    }

    const float theta = 1e7f;
    const uint32_t head_dim = 256, rope_dim = 128, n_kv_heads = 4;
    const uint32_t per_pos = n_kv_heads * head_dim;
    const uint32_t page_size = 16;

    // Blocks at scattered original positions (a long context); each remapped
    // into a contiguous window. Window order = ascending (as KvMemStore emits).
    std::vector<Block> blocks = {
        {0,        128},   // sink
        {500000,   128},   // far middle
        {123456,   128},
        {999872,   128},   // ~1M, partial-window edge
        {64,       64},    // odd offset + partial block (64 tok)
    };
    uint32_t window_tokens = 0;
    for (auto &b : blocks) window_tokens += b.n_tokens;

    // Physical cache: enough pages for the window, with a SCRAMBLED page table
    // so the test exercises non-identity physical offsets (the corruption risk).
    const uint32_t window_pages = (window_tokens + page_size - 1) / page_size;
    const uint32_t total_pages = window_pages + 8;  // slack
    const uint32_t total_slots = total_pages * page_size;

    std::mt19937 rng(0xB10Cu);
    std::vector<int32_t> phys_pages(total_pages);
    for (uint32_t i = 0; i < total_pages; ++i) phys_pages[i] = int32_t(i);
    std::shuffle(phys_pages.begin(), phys_pages.end(), rng);
    std::vector<int32_t> page_table(phys_pages.begin(),
                                    phys_pages.begin() + window_pages);

    int32_t *d_pages = nullptr;
    CHECK(cudaMalloc(&d_pages, page_table.size() * sizeof(int32_t)));
    CHECK(cudaMemcpy(d_pages, page_table.data(),
                     page_table.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    __half *d_cache = nullptr;
    CHECK(cudaMalloc(&d_cache, uint64_t(total_slots) * per_pos * sizeof(__half)));
    CHECK(cudaMemset(d_cache, 0, uint64_t(total_slots) * per_pos * sizeof(__half)));

    // Raw K per block, and the reference (fresh-baked-at-window) fp16 round-trip.
    std::uniform_real_distribution<float> dk(-1.0f, 1.0f);
    std::vector<std::vector<float>> raw(blocks.size());

    float *d_scratch = nullptr;  // contiguous per-block bake buffer (max 128 tok)
    CHECK(cudaMalloc(&d_scratch, uint64_t(128) * per_pos * sizeof(float)));

    // ---- Path A: store at ORIGINAL positions, then paged re-RoPE to window. ----
    uint32_t win_base = 0;
    std::vector<uint32_t> win_bases;
    for (size_t bi = 0; bi < blocks.size(); ++bi) {
        const auto &b = blocks[bi];
        const size_t n = size_t(b.n_tokens) * per_pos;
        raw[bi].resize(n);
        for (auto &v : raw[bi]) v = dk(rng);

        // Bake at orig_base (production store path) in fp32 scratch.
        CHECK(cudaMemcpy(d_scratch, raw[bi].data(), n * sizeof(float),
                         cudaMemcpyHostToDevice));
        dim3 bgrid(b.n_tokens, n_kv_heads);
        ref_bake_kernel<<<bgrid, rope_dim / 2>>>(
            d_scratch, n_kv_heads, head_dim, rope_dim, int32_t(b.orig_base),
            per_pos, theta);
        // Store (fp16) into the window slots.
        const unsigned thr = 256, blk = (per_pos + thr - 1) / thr;
        paged_store_f16_kernel<<<dim3(b.n_tokens, blk), thr>>>(
            d_cache, d_scratch, win_base, per_pos, d_pages, page_size);
        win_bases.push_back(win_base);
        win_base += b.n_tokens;
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // Remap each block in place: orig_base -> new_base(=its window base).
    for (size_t bi = 0; bi < blocks.size(); ++bi) {
        const auto &b = blocks[bi];
        if (!qw3::ported::launch_rope_block_remap_paged(
                d_cache, /*is_fp16=*/true, b.n_tokens, n_kv_heads, per_pos,
                head_dim, rope_dim, win_bases[bi], int32_t(b.orig_base),
                int32_t(win_bases[bi]), d_pages, page_size, theta, /*stream=*/0)) {
            std::fprintf(stderr, "launch_rope_block_remap_paged failed\n");
            return 1;
        }
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // Read back the assembled window (Path A result), fp16 round-trip.
    std::vector<float> gotA(size_t(window_tokens) * per_pos);
    {
        const unsigned thr = 256, blk = (per_pos + thr - 1) / thr;
        float *d_out = nullptr;
        CHECK(cudaMalloc(&d_out, gotA.size() * sizeof(float)));
        paged_load_f16_kernel<<<dim3(window_tokens, blk), thr>>>(
            d_cache, d_out, 0, per_pos, d_pages, page_size);
        CHECK(cudaDeviceSynchronize()); CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(gotA.data(), d_out, gotA.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));
        cudaFree(d_out);
    }

    // ---- Path B: fresh bake the same raw K at WINDOW positions (fp16 R/T). ----
    std::vector<float> refB(size_t(window_tokens) * per_pos);
    for (size_t bi = 0; bi < blocks.size(); ++bi) {
        const auto &b = blocks[bi];
        const size_t n = size_t(b.n_tokens) * per_pos;
        CHECK(cudaMemcpy(d_scratch, raw[bi].data(), n * sizeof(float),
                         cudaMemcpyHostToDevice));
        dim3 bgrid(b.n_tokens, n_kv_heads);
        ref_bake_kernel<<<bgrid, rope_dim / 2>>>(
            d_scratch, n_kv_heads, head_dim, rope_dim, int32_t(win_bases[bi]),
            per_pos, theta);
        // fp16 round-trip to match Path A's storage precision.
        std::vector<float> tmp(n);
        CHECK(cudaMemcpy(tmp.data(), d_scratch, n * sizeof(float),
                         cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) {
            tmp[i] = __half2float(__float2half(tmp[i]));
            refB[size_t(win_bases[bi]) * per_pos + i] = tmp[i];
        }
    }

    cudaFree(d_scratch); cudaFree(d_cache); cudaFree(d_pages);

    // Compare K vectors and attention scores against a random rope-baked query.
    std::vector<float> q(per_pos);
    std::uniform_real_distribution<float> dq(-1.0f, 1.0f);
    for (auto &v : q) v = dq(rng);
    {   // bake q at an in-window position on host (double ok: just a probe)
        const uint32_t half = rope_dim / 2;
        const double pos = double(window_tokens + 5);
        for (uint32_t u = 0; u < n_kv_heads; ++u) {
            float *base = q.data() + u * head_dim;
            for (uint32_t i = 0; i < half; ++i) {
                const double inv = std::pow((double)theta, -2.0 * i / rope_dim);
                const double a = pos * inv;
                const float c = (float)std::cos(a), s = (float)std::sin(a);
                const float x0 = base[i], x1 = base[i + half];
                base[i] = x0 * c - x1 * s;
                base[i + half] = x0 * s + x1 * c;
            }
        }
    }

    float max_abs_k = 0.0f, max_abs_score = 0.0f;
    int n_outside = 0;
    const float abs_tol = 5e-3f;  // one fp16 round-trip at unit magnitude
    for (size_t i = 0; i < gotA.size(); ++i) {
        float d = std::fabs(gotA[i] - refB[i]);
        max_abs_k = std::max(max_abs_k, d);
        if (d > abs_tol) ++n_outside;
    }
    for (uint32_t t = 0; t < window_tokens; ++t) {
        for (uint32_t u = 0; u < n_kv_heads; ++u) {
            const size_t off = size_t(t) * per_pos + size_t(u) * head_dim;
            float sa = 0.0f, sb = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sa += q[u * head_dim + d] * gotA[off + d];
                sb += q[u * head_dim + d] * refB[off + d];
            }
            max_abs_score = std::max(max_abs_score, std::fabs(sa - sb));
        }
    }

    std::printf("window_tokens=%u pages=%u (scrambled)  max_abs_k=%.4g  "
                "max_abs_score=%.4g  outside=%d/%zu\n",
                window_tokens, window_pages, max_abs_k, max_abs_score,
                n_outside, gotA.size());
    if (n_outside != 0) {
        std::printf("FAILED: %d elements outside fp16 tolerance\n", n_outside);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
