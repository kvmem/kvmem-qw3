// Batched re-RoPE block-remap parity test.
//
// The kvmem assembly step re-RoPEs every moved block from its baked position to
// its new in-window slot. The original path issued one launch per
// (standard-attention-layer x moved-block) — a launch storm. The batched kernel
// (rope_block_remap_paged_batched) collapses all moved blocks of one layer into
// a single launch whose grid covers (max_n_tokens, n_kv_heads, n_blocks).
//
// This test asserts the batched kernel is BIT-IDENTICAL to the per-block scalar
// kernel (launch_rope_block_remap_paged) it replaces. Both write each output
// element exactly once with the same FP ops and the same de-rotate/re-rotate
// math, and moved blocks occupy disjoint physical pages, so the result must be
// byte-equal (not merely within tolerance). The test exercises the partial-block
// guard (a short tail block) and the from==to skip (a block already at its slot).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
bool launch_rope_block_remap_paged_batched(void *cache, bool is_fp16,
                                           uint32_t n_blocks,
                                           uint32_t max_n_tokens,
                                           uint32_t n_kv_heads,
                                           uint32_t per_pos_size,
                                           uint32_t head_dim, uint32_t rope_dim,
                                           const int32_t *to_base,
                                           const int32_t *from_base,
                                           const int32_t *n_tokens,
                                           const int32_t *page_indices,
                                           uint32_t page_size, float theta,
                                           cudaStream_t stream);
}}  // namespace qw3::ported

#define CHECK(call) do {                                                  \
    cudaError_t _err = (call);                                            \
    if (_err != cudaSuccess) {                                            \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",                    \
                     __FILE__, __LINE__, cudaGetErrorString(_err));       \
        std::exit(1);                                                     \
    }                                                                     \
} while (0)

int main() {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::fprintf(stderr, "no CUDA devices, skipping\n");
        return 0;
    }

    // Qwen 3.6 attention geometry.
    const uint32_t n_kv_heads = 4;
    const uint32_t head_dim = 256;
    const uint32_t rope_dim = 128;       // partial RoPE over first half
    const uint32_t per_pos = n_kv_heads * head_dim;
    const uint32_t page_size = 16;
    const float theta = 1e7f;

    // Moved blocks: full 128-token blocks packed contiguously from window pos 0,
    // plus a from==to skip block and a short tail block to hit both guards.
    struct Blk { int32_t to_base; int32_t from_base; int32_t n_tokens; };
    std::vector<Blk> blks = {
        {   0,  100000, 128},
        { 128,  523777, 128},
        { 256,     256, 128},   // from == to: skip (no write in either path)
        { 384,  938210, 128},
        { 512,       7, 128},
        { 640,  271828, 128},
        { 768,  141421, 128},
        { 896,  999999,  64},   // partial tail block: exercises tok-bound guard
    };
    const uint32_t n_blocks = static_cast<uint32_t>(blks.size());
    uint32_t max_n_tokens = 0;
    for (auto &b : blks) max_n_tokens = std::max<uint32_t>(max_n_tokens, b.n_tokens);

    // Window page table: window page p -> a scattered, disjoint physical page.
    // Window spans 7*128 + 64 = 960 tokens => 60 pages.
    const uint32_t n_window_tokens = 7 * 128 + 64;
    const uint32_t n_window_pages = (n_window_tokens + page_size - 1) / page_size;
    std::vector<int32_t> page_indices(n_window_pages);
    int32_t max_phys = 0;
    for (uint32_t p = 0; p < n_window_pages; ++p) {
        page_indices[p] = static_cast<int32_t>(p * 2 + 3);  // disjoint, non-identity
        max_phys = std::max(max_phys, page_indices[p]);
    }
    const uint64_t n_positions = static_cast<uint64_t>(max_phys + 1) * page_size;
    const uint64_t n_elems = n_positions * per_pos;

    // Random fp16 cache contents at the referenced physical positions.
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__half> host(n_elems);
    for (auto &h : host) h = __float2half(dist(rng));

    // Two device copies of the identical input.
    __half *d_ref = nullptr;
    __half *d_cand = nullptr;
    CHECK(cudaMalloc(&d_ref, n_elems * sizeof(__half)));
    CHECK(cudaMalloc(&d_cand, n_elems * sizeof(__half)));
    CHECK(cudaMemcpy(d_ref, host.data(), n_elems * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_cand, host.data(), n_elems * sizeof(__half), cudaMemcpyHostToDevice));

    int32_t *d_pages = nullptr;
    CHECK(cudaMalloc(&d_pages, n_window_pages * sizeof(int32_t)));
    CHECK(cudaMemcpy(d_pages, page_indices.data(), n_window_pages * sizeof(int32_t),
                     cudaMemcpyHostToDevice));

    // Reference: per-block scalar launches (skip from==to, mirroring the
    // production loop that used `if (rm.skip) continue;`).
    for (auto &b : blks) {
        if (b.from_base == b.to_base) continue;
        if (!qw3::ported::launch_rope_block_remap_paged(
                d_ref, /*is_fp16=*/true, static_cast<uint32_t>(b.n_tokens),
                n_kv_heads, per_pos, head_dim, rope_dim,
                static_cast<uint32_t>(b.to_base), b.from_base, b.to_base,
                d_pages, page_size, theta, /*stream=*/0)) {
            std::fprintf(stderr, "scalar launch failed\n");
            return 1;
        }
    }
    CHECK(cudaDeviceSynchronize());

    // Candidate: one batched launch over all blocks (the kernel internally
    // skips from==to via its orig_base==new_base guard and bounds tok per block).
    std::vector<int32_t> to_base, from_base, ntok;
    for (auto &b : blks) {
        to_base.push_back(b.to_base);
        from_base.push_back(b.from_base);
        ntok.push_back(b.n_tokens);
    }
    int32_t *d_to = nullptr, *d_from = nullptr, *d_ntok = nullptr;
    CHECK(cudaMalloc(&d_to, n_blocks * sizeof(int32_t)));
    CHECK(cudaMalloc(&d_from, n_blocks * sizeof(int32_t)));
    CHECK(cudaMalloc(&d_ntok, n_blocks * sizeof(int32_t)));
    CHECK(cudaMemcpy(d_to, to_base.data(), n_blocks * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_from, from_base.data(), n_blocks * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ntok, ntok.data(), n_blocks * sizeof(int32_t), cudaMemcpyHostToDevice));

    if (!qw3::ported::launch_rope_block_remap_paged_batched(
            d_cand, /*is_fp16=*/true, n_blocks, max_n_tokens, n_kv_heads,
            per_pos, head_dim, rope_dim, d_to, d_from, d_ntok, d_pages,
            page_size, theta, /*stream=*/0)) {
        std::fprintf(stderr, "batched launch failed\n");
        return 1;
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    std::vector<__half> ref(n_elems), cand(n_elems);
    CHECK(cudaMemcpy(ref.data(), d_ref, n_elems * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cand.data(), d_cand, n_elems * sizeof(__half), cudaMemcpyDeviceToHost));

    uint64_t mismatches = 0;
    float max_abs = 0.0f;
    for (uint64_t i = 0; i < n_elems; ++i) {
        const uint16_t rb = *reinterpret_cast<const uint16_t *>(&ref[i]);
        const uint16_t cb = *reinterpret_cast<const uint16_t *>(&cand[i]);
        if (rb != cb) {
            ++mismatches;
            float d = std::fabs(__half2float(ref[i]) - __half2float(cand[i]));
            max_abs = std::max(max_abs, d);
        }
    }

    std::printf("batched vs scalar paged re-RoPE: elems=%llu blocks=%u "
                "max_n_tokens=%u  bit_mismatches=%llu  max_abs=%.4g\n",
                static_cast<unsigned long long>(n_elems), n_blocks,
                max_n_tokens, static_cast<unsigned long long>(mismatches),
                max_abs);

    cudaFree(d_ref); cudaFree(d_cand); cudaFree(d_pages);
    cudaFree(d_to); cudaFree(d_from); cudaFree(d_ntok);

    if (mismatches != 0) {
        std::printf("FAILED: batched kernel diverges from scalar (%llu elems)\n",
                    static_cast<unsigned long long>(mismatches));
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
