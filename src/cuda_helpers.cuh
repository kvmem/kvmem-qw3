// SPDX-License-Identifier: MIT
//
// Generic CUDA helpers shared by ported kernels. Primitives lifted from
// llama.cpp ggml/src/ggml-cuda/common.cuh @ commit 57ebaf4
// (see LICENSES/llama.cpp.txt). Stripped to the minimum qw3 needs.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace qw3 {
namespace cuda_helpers {

// Q8_0 weights are stored on-device in a SPLIT-PLANE row layout (v7+):
//
//   row_base[r] = base + r * n_blocks * 34
//   ┌─────────────────────────────────────────┐
//   │ Plane S (FP16 scales): n_blocks × 2 B   │  16-byte aligned (n_blocks % 16 == 0)
//   ├─────────────────────────────────────────┤
//   │ Plane Q (int8 quants): n_blocks × 32 B  │  starts at byte (2*n_blocks), 16-byte aligned
//   └─────────────────────────────────────────┘
//
// vs the legacy on-disk Q8_0 layout (interleaved 34-byte blocks: [d; qs[32]]).
// Total bytes per row are unchanged (34 * n_blocks). The split-plane layout
// places qs[] on a 16-byte boundary so cp.async.cg.shared.global with 16-byte
// transfers becomes legal — required for the v7 pipelined MMQ kernel.
//
// All Q8_0 readers in kernels_cuda.cu / mmvq_q8.cu / mmq_q8.cu use these
// accessors. Repack happens once at weight upload (CudaWeight Q8_0 ctor).
__device__ __host__ __forceinline__ const __half *q8_d_plane(const uint8_t *row_base) {
    return reinterpret_cast<const __half *>(row_base);
}

__device__ __host__ __forceinline__ const int8_t *q8_qs_plane(const uint8_t *row_base,
                                                              uint64_t blocks_per_row) {
    return reinterpret_cast<const int8_t *>(row_base + 2 * blocks_per_row);
}

// Warp-level sum reduction. width is the active lane count (must be a power of
// two divisor of 32). On Ampere+ for int the compiler can lower this to
// __reduce_add_sync; for float we always use __shfl_xor_sync.
template <int width = 32>
__device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, offset, width);
    }
    return x;
}

// Warp-level max reduction. Same width semantics as warp_reduce_sum.
template <int width = 32>
__device__ __forceinline__ float warp_reduce_max(float x) {
    #pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, offset, width));
    }
    return x;
}

// Pre-computed magic numbers for fast unsigned division-by-constant on the
// device. n / d  is computed as (mulhi(n, mp) + n) >> L. Pack the divisor in
// .z so callers can recover it.
inline uint3 init_fastdiv_values(uint32_t d) {
    // ceil(log2(d))
    uint32_t L = 0;
    while (L < 32 && (uint32_t{1} << L) < d) {
        ++L;
    }
    uint32_t mp = static_cast<uint32_t>(
        ((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d) + 1);
    return make_uint3(mp, L, d);
}

__device__ __forceinline__ uint32_t fastdiv(uint32_t n, uint3 fd) {
    const uint32_t hi = __umulhi(n, fd.x);
    return (hi + n) >> fd.y;
}

__device__ __forceinline__ uint32_t fastmodulo(uint32_t n, uint3 fd) {
    return n - fastdiv(n, fd) * fd.z;
}

} // namespace cuda_helpers
} // namespace qw3
