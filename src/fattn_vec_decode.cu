// SPDX-License-Identifier: MIT
//
// Flash-attention-style decode kernel — single token query, vector-style.
//
// Inspired by llama.cpp ggml/src/ggml-cuda/fattn-vec.cuh @ commit
// 57ebaf4edd99ea675f256ae2286cd99206dbfcd1 (MIT, see LICENSES/llama.cpp.txt).
//
// What we kept from upstream:
//   - Parallelization pattern: each warp handles a different KV token slot
//     (so dot products across tokens run in parallel, not sequentially).
//   - Per-thread V accumulator held in registers across the whole KV stream;
//     online softmax keeps a running (max, sum, acc).
//   - Two-stage reduction: warp-level shuffle for K·Q, shared-memory tile
//     for cross-warp softmax.
//
// What we stripped (qw3 doesn't need any of it for Qwen 3.6 27B decode):
//   - Quantized K / V caches (we use F32; ggml's mmq variants gone).
//   - Multi-column / batched-Q (decode is always 1 token here).
//   - ALiBi slopes, logit-softcap, attention sinks.
//   - Mask tensor (causal is implicit: t < seq_len).
//   - F16/BF16 dispatch hairball — pure F32 K/V.
//   - The launch_fattn helper and dst_meta machinery. Output is written
//     directly to dst with shape [n_heads, head_dim].
//
// The kernel signature matches the existing attention_decode_fused_kernel
// well enough that the dispatcher can pick between them at runtime.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>
#include <math.h>
#include <type_traits>

#include "cuda_helpers.cuh"

namespace qw3 {
namespace ported {

// Type-dispatched K/V load. The kernel templates on KVT so the same code
// drives FP32 and FP16 caches; the only per-type bit is how a single
// element is read into a float.
template <typename T>
__device__ __forceinline__ float kv_load(const T *p);

template <>
__device__ __forceinline__ float kv_load<float>(const float *p) { return *p; }

template <>
__device__ __forceinline__ float kv_load<__half>(const __half *p) { return __half2float(*p); }

// One CUDA block per (head, batch). 128 threads = 4 warps. Each warp owns one
// KV-token slot per outer-iteration, so 4 KV tokens are scored concurrently.
//
// Each thread holds a V_FRAGMENT-sized slice of the running V accumulator in
// registers. With D=128 and 32 lanes per warp, each thread carries D/32 = 4
// floats. 4 warps would together hold the whole D vector 4 times; we instead
// have only WARP 0 own the running accumulator and let other warps feed
// softmax-weighted V slices into it via shared memory.
//
// More concretely:
//   - Outer loop: tiles of NWARPS KV tokens at a time.
//   - Each warp scores its assigned token (one warp_reduce_sum across 32
//     lanes that each hold D/32 floats of K and Q). The score lands in
//     shmem at s_scores[warp].
//   - All warps sync. Softmax is updated using the online-softmax recurrence
//     (the same one as the existing attention_decode_fused_kernel).
//   - V update: each warp reads its assigned token's V, scales by its
//     softmax weight, and atomically adds into a shared accumulator (or, for
//     the case where each warp writes a different DK shard, no atomic).
//
// We use the second pattern: each lane owns D/32 elements of the V accumulator
// (so 32 lanes cover D). Warp 0's lanes own the canonical accumulator slice.
// At the end of each outer iteration, warps 1..NWARPS-1 stage their weighted
// contributions in shmem and warp 0 reduces them in.

template <uint32_t HEAD_DIM, typename KVT, uint32_t NSPLIT>
__global__ void
__launch_bounds__(128, 1)
fattn_vec_decode_kernel(
        float       * __restrict__ out,         // [batch, n_heads, HEAD_DIM] (when NSPLIT==1)
                                                // partial buf [n_heads, batch, NSPLIT, HEAD_DIM] otherwise
        float       * __restrict__ ms_buf,      // [n_heads, batch, NSPLIT, 2] (max,sum) — only used when NSPLIT>1
        const float * __restrict__ q,           // [batch, n_heads, q_stride] (gate after head_dim)
        uint32_t     q_stride,
        const KVT   * __restrict__ k_cache,     // [seq, n_kv_heads, HEAD_DIM]
        const KVT   * __restrict__ v_cache,     // [seq, n_kv_heads, HEAD_DIM]
        uint32_t     n_heads,
        uint32_t     n_kv_heads,
        uint32_t     base_seq_len,
        uint32_t     q_batch_stride,
        uint32_t     out_batch_stride,
        float        scale) {
    constexpr uint32_t NWARPS    = 4;
    constexpr uint32_t WARP_SIZE = 32;
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be a multiple of warp size");
    constexpr uint32_t D_PER_LANE = HEAD_DIM / WARP_SIZE;   // e.g. 128/32 = 4

    const uint32_t head    = blockIdx.x;
    const uint32_t b       = blockIdx.y;
    const uint32_t split   = blockIdx.z;
    const uint32_t tid     = threadIdx.x;
    const uint32_t warp    = tid / WARP_SIZE;
    const uint32_t lane    = tid % WARP_SIZE;
    if (head >= n_heads) return;

    const uint32_t seq_len = base_seq_len + b + 1;
    const uint32_t kv_head = head / (n_heads / n_kv_heads);

    // Split-K: this block handles tokens [t_lo, t_hi).
    const uint32_t per_split = (seq_len + NSPLIT - 1) / NSPLIT;
    const uint32_t t_lo      = split * per_split;
    const uint32_t t_hi      = (t_lo + per_split < seq_len) ? (t_lo + per_split) : seq_len;

    // Q held in registers (each thread carries D_PER_LANE floats; warps share
    // the same Q so we just have every warp load it from shmem).
    __shared__ float s_q[HEAD_DIM];
    const float * q_ptr = q + static_cast<uint64_t>(b) * q_batch_stride;
    if (warp == 0) {
        // Warp 0 lanes load HEAD_DIM elements collaboratively.
        #pragma unroll
        for (uint32_t r = 0; r < D_PER_LANE; ++r) {
            s_q[r * WARP_SIZE + lane] = q_ptr[head * q_stride + r * WARP_SIZE + lane];
        }
    }
    __syncthreads();

    // Per-lane Q slice in registers — same offsets each lane will use for K.
    float q_reg[D_PER_LANE];
    #pragma unroll
    for (uint32_t r = 0; r < D_PER_LANE; ++r) {
        q_reg[r] = s_q[r * WARP_SIZE + lane];
    }

    // Running softmax + running V accumulator.
    // VKQ[r] is the partial sum for the r-th D-shard owned by this lane.
    // Only lane-r-of-warp-0's slice matters for the final write, but all
    // lanes accumulate identically (each lane covers a unique slice of D).
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float VKQ[D_PER_LANE] = {0.0f};

    __shared__ float s_scores[NWARPS];

    for (uint32_t t0 = t_lo; t0 < t_hi; t0 += NWARPS) {
        const uint32_t t = t0 + warp;
        const bool active = (t < t_hi);

        // 1) Score K[t] · Q.
        float local = 0.0f;
        if (active) {
            const KVT * k_t = k_cache + static_cast<uint64_t>(t) * n_kv_heads * HEAD_DIM
                              + kv_head * HEAD_DIM;
            #pragma unroll
            for (uint32_t r = 0; r < D_PER_LANE; ++r) {
                local += q_reg[r] * kv_load<KVT>(k_t + r * WARP_SIZE + lane);
            }
        }
        local = cuda_helpers::warp_reduce_sum<WARP_SIZE>(local);
        if (lane == 0) {
            s_scores[warp] = active ? (local * scale) : -INFINITY;
        }
        __syncthreads();

        // 2) Online softmax update — every thread does the same arithmetic
        //    so we can avoid another sync.
        float tile_max = -INFINITY;
        #pragma unroll
        for (uint32_t w = 0; w < NWARPS; ++w) {
            const float v = s_scores[w];
            if (v > tile_max) tile_max = v;
        }
        const float new_max = fmaxf(running_max, tile_max);
        const float prev_scale = (running_max == -INFINITY) ? 0.0f
                                                            : __expf(running_max - new_max);
        running_sum *= prev_scale;
        #pragma unroll
        for (uint32_t r = 0; r < D_PER_LANE; ++r) VKQ[r] *= prev_scale;

        // 3) Accumulate V[t] weighted by softmax(score - new_max).
        //    Each warp pulls its own t's V row; lane r reads the lane-shard.
        if (active) {
            const float w_t = __expf(s_scores[warp] - new_max);
            // Sum the weights across warps to update running_sum. Single lane
            // per warp publishes; warp 0 lane 0 collects.
            // (We can compute sum across all 4 warps by reading s_scores[],
            // but it's simpler to just have each lane independently
            // accumulate the per-warp contribution into running_sum.)
            // Every thread updates running_sum identically — same s_scores[]
            // visible to all.
            // (Done below outside the active block to keep arithmetic uniform.)

            const KVT * v_t = v_cache + static_cast<uint64_t>(t) * n_kv_heads * HEAD_DIM
                              + kv_head * HEAD_DIM;
            #pragma unroll
            for (uint32_t r = 0; r < D_PER_LANE; ++r) {
                VKQ[r] += w_t * kv_load<KVT>(v_t + r * WARP_SIZE + lane);
            }
        }

        // Update running_sum from all NWARPS scores (uniform on every thread).
        #pragma unroll
        for (uint32_t w = 0; w < NWARPS; ++w) {
            const float s = s_scores[w];
            if (s != -INFINITY) {
                running_sum += __expf(s - new_max);
            }
        }

        running_max = new_max;
        __syncthreads();    // s_scores reuse next iteration
    }

    // Cross-warp combine. Each lane has a partial VKQ in its slice (D shard
    // r*WARP_SIZE+lane). Warp 0 owns the final write; warps 1..NWARPS-1
    // stage their per-lane contributions through shmem.
    __shared__ float s_combine[NWARPS - 1][HEAD_DIM];
    if (warp > 0) {
        #pragma unroll
        for (uint32_t r = 0; r < D_PER_LANE; ++r) {
            s_combine[warp - 1][r * WARP_SIZE + lane] = VKQ[r];
        }
    }
    __syncthreads();
    if (warp == 0) {
        // Empty slice (t_lo >= t_hi): can happen for trailing splits when
        // seq_len < NSPLIT * per_split. Mark as inactive and stop.
        const bool slice_empty = (t_hi <= t_lo);
        #pragma unroll
        for (uint32_t r = 0; r < D_PER_LANE; ++r) {
            float acc = VKQ[r];
            #pragma unroll
            for (uint32_t w = 0; w < NWARPS - 1; ++w) {
                acc += s_combine[w][r * WARP_SIZE + lane];
            }
            // For NSPLIT==1 we have the full numerator and can normalize
            // here. For NSPLIT>1 we hand off the un-normalized partial
            // numerator + (max, sum) pair to the combiner kernel.
            if (NSPLIT == 1) {
                float * out_ptr = out + static_cast<uint64_t>(b) * out_batch_stride;
                out_ptr[head * HEAD_DIM + r * WARP_SIZE + lane] = acc / running_sum;
            } else {
                const uint64_t base = (static_cast<uint64_t>(head) * gridDim.y + b)
                                      * NSPLIT + split;
                float * partial = out + base * HEAD_DIM;
                partial[r * WARP_SIZE + lane] = slice_empty ? 0.0f : acc;
            }
        }
        if (NSPLIT > 1 && lane == 0) {
            const uint64_t base = (static_cast<uint64_t>(head) * gridDim.y + b)
                                  * NSPLIT + split;
            ms_buf[base * 2 + 0] = slice_empty ? -INFINITY : running_max;
            ms_buf[base * 2 + 1] = slice_empty ? 0.0f      : running_sum;
        }
    }
}

// Combine NSPLIT partial (max, sum, VKQ_un_normalized) tuples per (head, batch)
// into the final output via online softmax. One block per (head, batch),
// 32 lanes (one warp) is enough for n_heads up to a few hundred.
template <uint32_t HEAD_DIM, uint32_t NSPLIT>
__global__ void
__launch_bounds__(HEAD_DIM, 2)
fattn_vec_combine_kernel(
        float       * __restrict__ out,
        const float * __restrict__ partials,    // [n_heads, batch, NSPLIT, HEAD_DIM]
        const float * __restrict__ ms,          // [n_heads, batch, NSPLIT, 2]
        uint32_t     n_heads,
        uint32_t     batch,
        uint32_t     out_batch_stride) {
    const uint32_t head = blockIdx.x;
    const uint32_t b    = blockIdx.y;
    const uint32_t d    = threadIdx.x;
    if (head >= n_heads || d >= HEAD_DIM) return;

    const uint64_t mb_base = (static_cast<uint64_t>(head) * batch + b) * NSPLIT;

    // First pass: find global max across slices (in shared memory).
    __shared__ float s_max[NSPLIT];
    __shared__ float s_sum[NSPLIT];
    if (d < NSPLIT) {
        s_max[d] = ms[(mb_base + d) * 2 + 0];
        s_sum[d] = ms[(mb_base + d) * 2 + 1];
    }
    __syncthreads();

    float gmax = -INFINITY;
    #pragma unroll
    for (uint32_t s = 0; s < NSPLIT; ++s) {
        if (s_sum[s] > 0.0f && s_max[s] > gmax) gmax = s_max[s];
    }

    // Now combine numerator and denominator:
    // gsum = Σ_s sum_s * exp(max_s - gmax)
    // gnum[d] = Σ_s VKQ_s[d] * exp(max_s - gmax)
    float gsum  = 0.0f;
    float gnum  = 0.0f;
    #pragma unroll
    for (uint32_t s = 0; s < NSPLIT; ++s) {
        if (s_sum[s] <= 0.0f) continue;
        const float w = __expf(s_max[s] - gmax);
        gsum += s_sum[s] * w;
        const uint64_t pbase = (mb_base + s) * HEAD_DIM;
        gnum += partials[pbase + d] * w;
    }

    float * out_ptr = out + static_cast<uint64_t>(b) * out_batch_stride;
    out_ptr[head * HEAD_DIM + d] = gnum / gsum;
}

// Pick NSPLIT based on seq_len so we put more blocks on the GPU when KV
// gets large. Below ~1K tokens the launch overhead and combiner kernel
// outweigh the parallelism gain; above ~2K tokens 8-way split fills SMs.
static inline uint32_t pick_nsplit(uint32_t seq_len) {
    if (seq_len <= 1024) return 1;
    if (seq_len <= 2048) return 2;
    if (seq_len <= 4096) return 4;
    return 8;
}

// Scratch sizing helper (in bytes). Caller must allocate at least this many
// bytes for both `partials` and `ms` (combined or separate, doesn't matter).
size_t fattn_vec_scratch_bytes(uint32_t n_heads, uint32_t batch,
                               uint32_t head_dim, uint32_t seq_len) {
    const uint32_t nsplit = pick_nsplit(seq_len);
    if (nsplit == 1) return 0;
    const size_t partials = static_cast<size_t>(n_heads) * batch * nsplit * head_dim
                          * sizeof(float);
    const size_t ms       = static_cast<size_t>(n_heads) * batch * nsplit * 2
                          * sizeof(float);
    return partials + ms;
}

// Internal launch helper: dispatches on (HEAD_DIM, KVT, NSPLIT). Caller has
// already chosen NSPLIT based on seq_len. partials/ms are required when
// NSPLIT > 1; ignored otherwise.
template <typename KVT>
static bool dispatch_launch(
        float * out,
        float * partials,
        float * ms,
        const float * q,
        uint32_t q_stride,
        const KVT * k_cache,
        const KVT * v_cache,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t seq_len,
        uint32_t batch,
        uint32_t q_batch_stride,
        uint32_t out_batch_stride,
        float    scale,
        uint32_t nsplit,
        cudaStream_t stream) {
    const dim3 block(128);
    const uint32_t base_seq_len = seq_len - 1;
    const uint32_t b_eff = batch == 0 ? 1 : batch;

    if (head_dim != 128 && head_dim != 256) return false;

    auto launch = [&](auto HD_C, auto NS_C) {
        constexpr uint32_t HD = decltype(HD_C)::value;
        constexpr uint32_t NS = decltype(NS_C)::value;
        const dim3 grid(n_heads, b_eff, NS);
        // For NSPLIT==1 the partial pointer is unused — pass `out` as
        // dummy to satisfy the kernel's pointer-arithmetic on the
        // NSPLIT>1 branch (which won't be taken).
        float * dst = (NS == 1) ? out : partials;
        fattn_vec_decode_kernel<HD, KVT, NS><<<grid, block, 0, stream>>>(
            dst, ms, q, q_stride, k_cache, v_cache,
            n_heads, n_kv_heads, base_seq_len,
            q_batch_stride, out_batch_stride, scale);
        if (NS > 1) {
            const dim3 cgrid(n_heads, b_eff, 1);
            const dim3 cblock(HD);
            fattn_vec_combine_kernel<HD, NS><<<cgrid, cblock, 0, stream>>>(
                out, partials, ms, n_heads, b_eff, out_batch_stride);
        }
    };

    #define DISPATCH(HD, NS)                                              \
        if (head_dim == HD && nsplit == NS) {                             \
            launch(std::integral_constant<uint32_t, HD>{},                \
                   std::integral_constant<uint32_t, NS>{});               \
            return true;                                                  \
        }
    DISPATCH(128, 1) DISPATCH(128, 2) DISPATCH(128, 4) DISPATCH(128, 8)
    DISPATCH(256, 1) DISPATCH(256, 2) DISPATCH(256, 4) DISPATCH(256, 8)
    #undef DISPATCH
    return false;
}

// Host-side launcher. Returns true on success; false if HEAD_DIM unsupported.
bool launch_fattn_vec_decode_f32(
        float *       out,
        const float * q,
        uint32_t      q_stride,
        const float * k_cache,
        const float * v_cache,
        uint32_t      n_heads,
        uint32_t      n_kv_heads,
        uint32_t      head_dim,
        uint32_t      seq_len,
        uint32_t      batch,            // typically 1
        uint32_t      q_batch_stride,
        uint32_t      out_batch_stride,
        float         scale,
        cudaStream_t  stream) {
    return dispatch_launch<float>(
        out, /*partials=*/nullptr, /*ms=*/nullptr,
        q, q_stride, k_cache, v_cache,
        n_heads, n_kv_heads, head_dim, seq_len, batch,
        q_batch_stride, out_batch_stride, scale,
        /*nsplit=*/1, stream);
}

// Split-K variant. Caller passes a scratch buffer at least
// fattn_vec_scratch_bytes() large; first half holds VKQ partials, second
// holds (max, sum) tuples.
bool launch_fattn_vec_decode_f32_splitk(
        float *       out,
        void *        scratch,
        const float * q,
        uint32_t      q_stride,
        const float * k_cache,
        const float * v_cache,
        uint32_t      n_heads,
        uint32_t      n_kv_heads,
        uint32_t      head_dim,
        uint32_t      seq_len,
        uint32_t      batch,
        uint32_t      q_batch_stride,
        uint32_t      out_batch_stride,
        float         scale,
        cudaStream_t  stream) {
    const uint32_t b_eff = batch == 0 ? 1 : batch;
    const uint32_t nsplit = pick_nsplit(seq_len);
    if (nsplit == 1) {
        return dispatch_launch<float>(
            out, nullptr, nullptr,
            q, q_stride, k_cache, v_cache,
            n_heads, n_kv_heads, head_dim, seq_len, batch,
            q_batch_stride, out_batch_stride, scale, /*nsplit=*/1, stream);
    }
    float * partials = static_cast<float *>(scratch);
    float * ms       = partials + static_cast<size_t>(n_heads) * b_eff * nsplit * head_dim;
    return dispatch_launch<float>(
        out, partials, ms,
        q, q_stride, k_cache, v_cache,
        n_heads, n_kv_heads, head_dim, seq_len, batch,
        q_batch_stride, out_batch_stride, scale, nsplit, stream);
}

// FP16 K/V cache variant. Q stays FP32 (it's already in registers and only
// (n_heads * head_dim) floats per token); converting Q to FP16 wouldn't save
// any read bandwidth at decode time. Reads are 2x cheaper than the F32 path.
bool launch_fattn_vec_decode_f16(
        float *       out,
        const float * q,
        uint32_t      q_stride,
        const void  * k_cache,             // __half *
        const void  * v_cache,             // __half *
        uint32_t      n_heads,
        uint32_t      n_kv_heads,
        uint32_t      head_dim,
        uint32_t      seq_len,
        uint32_t      batch,
        uint32_t      q_batch_stride,
        uint32_t      out_batch_stride,
        float         scale,
        cudaStream_t  stream) {
    return dispatch_launch<__half>(
        out, /*partials=*/nullptr, /*ms=*/nullptr,
        q, q_stride,
        static_cast<const __half *>(k_cache),
        static_cast<const __half *>(v_cache),
        n_heads, n_kv_heads, head_dim, seq_len, batch,
        q_batch_stride, out_batch_stride, scale, /*nsplit=*/1, stream);
}

bool launch_fattn_vec_decode_f16_splitk(
        float *       out,
        void *        scratch,
        const float * q,
        uint32_t      q_stride,
        const void  * k_cache,
        const void  * v_cache,
        uint32_t      n_heads,
        uint32_t      n_kv_heads,
        uint32_t      head_dim,
        uint32_t      seq_len,
        uint32_t      batch,
        uint32_t      q_batch_stride,
        uint32_t      out_batch_stride,
        float         scale,
        cudaStream_t  stream) {
    const uint32_t b_eff = batch == 0 ? 1 : batch;
    const uint32_t nsplit = pick_nsplit(seq_len);
    if (nsplit == 1) {
        return dispatch_launch<__half>(
            out, nullptr, nullptr, q, q_stride,
            static_cast<const __half *>(k_cache),
            static_cast<const __half *>(v_cache),
            n_heads, n_kv_heads, head_dim, seq_len, batch,
            q_batch_stride, out_batch_stride, scale, /*nsplit=*/1, stream);
    }
    float * partials = static_cast<float *>(scratch);
    float * ms       = partials + static_cast<size_t>(n_heads) * b_eff * nsplit * head_dim;
    return dispatch_launch<__half>(
        out, partials, ms, q, q_stride,
        static_cast<const __half *>(k_cache),
        static_cast<const __half *>(v_cache),
        n_heads, n_kv_heads, head_dim, seq_len, batch,
        q_batch_stride, out_batch_stride, scale, nsplit, stream);
}

} // namespace ported
} // namespace qw3
