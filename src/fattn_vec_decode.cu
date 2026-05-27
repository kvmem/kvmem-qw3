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
#include <cstdlib>
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

// ---------------------------------------------------------------------------
// Tiled flash-attention prefill kernel (qw3-side, FA2 style).
//
// Why this exists separately from fattn_vec_decode_kernel: the decode kernel
// is parametrized as one block per (head, query) and reads K/V independently
// per query, with no reuse across queries. For prefill, batch == T queries,
// and each query reads the full t < base + b + 1 KV stream. That is O(T²)
// HBM traffic with zero K/V reuse. nsys on a 4K prompt shows fattn_vec is
// ~534 ms of the 1.4 s prefill at 4096 tokens.
//
// Design (FA2 with BR=4, BC=32):
//   - Grid: (n_heads, ceil(batch / BR), 1).
//   - Block: 4 warps × 32 lanes = 128 threads.
//   - Each warp owns one query in the [block_q*BR, block_q*BR + BR) range.
//   - All 4 warps cooperatively load a BC=32 token tile of K and V into
//     shmem, then each warp scores and accumulates against its own query.
//   - K and V loaded into shmem ONCE per BR=4 queries → 4× HBM bandwidth
//     reduction on K/V vs the per-query decode kernel.
//   - Online softmax: per-warp running (max, sum, VKQ).
//   - Causal: scores at kv_idx > base + q_idx are masked to -INFINITY.
//
// Shmem (HEAD_DIM=256, FP16 K/V):
//   K tile: 32 × 256 × 2 = 16 KB
//   V tile: 32 × 256 × 2 = 16 KB
//   Total = 32 KB (fits in default 48 KB static shmem limit on every arch
//   we target — no cudaFuncSetAttribute needed).

template <uint32_t HEAD_DIM, typename KVT, uint32_t BR, uint32_t BC>
__global__ void
__launch_bounds__(BR * 32, 1)
fattn_prefill_kernel(
        float       * __restrict__ out,         // [batch, n_heads, HEAD_DIM]
        const float * __restrict__ q,           // [batch, n_heads, q_stride]
        uint32_t     q_stride,
        const KVT   * __restrict__ k_cache,     // [seq, n_kv_heads, HEAD_DIM]
        const KVT   * __restrict__ v_cache,
        uint32_t     n_heads,
        uint32_t     n_kv_heads,
        uint32_t     base_seq_len,
        uint32_t     batch,
        uint32_t     q_batch_stride,
        uint32_t     out_batch_stride,
        float        scale) {
    constexpr uint32_t WARP_SIZE  = 32;
    constexpr uint32_t NWARPS     = BR;
    constexpr uint32_t D_PER_LANE = HEAD_DIM / WARP_SIZE;
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be a multiple of warp size");

    const uint32_t head    = blockIdx.x;
    const uint32_t block_q = blockIdx.y;
    const uint32_t tid     = threadIdx.x;
    const uint32_t warp    = tid / WARP_SIZE;
    const uint32_t lane    = tid % WARP_SIZE;
    if (head >= n_heads) return;

    const uint32_t q_idx   = block_q * BR + warp;
    const bool warp_active = (q_idx < batch);
    const uint32_t kv_head = head / (n_heads / n_kv_heads);

    // The tile loop runs from 0 up to the maximum kv index this block needs.
    // Active queries see different causal limits, but we share one outer
    // bound so all warps do the same number of tile iterations.
    uint32_t block_max_q = block_q * BR + BR;
    if (block_max_q > batch) block_max_q = batch;
    const uint32_t block_max_kv = base_seq_len + block_max_q;   // exclusive
    const uint32_t my_max_kv    = base_seq_len + q_idx + 1;     // exclusive (causal)

    // Q in registers (per-warp).
    float q_reg[D_PER_LANE];
    if (warp_active) {
        const float * q_ptr = q + static_cast<uint64_t>(q_idx) * q_batch_stride
                                + head * q_stride;
        #pragma unroll
        for (uint32_t r = 0; r < D_PER_LANE; ++r) {
            q_reg[r] = q_ptr[r * WARP_SIZE + lane];
        }
    } else {
        #pragma unroll
        for (uint32_t r = 0; r < D_PER_LANE; ++r) q_reg[r] = 0.0f;
    }

    // Running softmax + V accumulator.
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float VKQ[D_PER_LANE] = {0.0f};

    // Shmem K/V tiles. KVT (= __half or float) per element.
    __shared__ KVT s_k[BC * HEAD_DIM];
    __shared__ KVT s_v[BC * HEAD_DIM];

    // Each tile: load BC tokens of K and V from HBM into shmem cooperatively.
    // Layout in shmem: s_k[t * HEAD_DIM + d], same for s_v. 128 threads
    // cooperate; each thread handles HEAD_DIM/(WARP_SIZE) lanes ≡ D_PER_LANE
    // dims per token, NWARPS tokens deep.
    static_assert(BC % NWARPS == 0, "BC must be a multiple of NWARPS");
    constexpr uint32_t TOKENS_PER_WARP = BC / NWARPS;

    for (uint32_t t0 = 0; t0 < block_max_kv; t0 += BC) {
        // Cooperative tile load. Warp `w` loads tokens [t0 + w * TPW, t0 + (w+1) * TPW).
        #pragma unroll
        for (uint32_t tw = 0; tw < TOKENS_PER_WARP; ++tw) {
            const uint32_t t = t0 + warp * TOKENS_PER_WARP + tw;
            const bool t_in_range = (t < block_max_kv);
            const uint32_t shmem_t = warp * TOKENS_PER_WARP + tw;
            const KVT * k_src = t_in_range
                ? (k_cache + static_cast<uint64_t>(t) * n_kv_heads * HEAD_DIM
                           + kv_head * HEAD_DIM)
                : nullptr;
            const KVT * v_src = t_in_range
                ? (v_cache + static_cast<uint64_t>(t) * n_kv_heads * HEAD_DIM
                           + kv_head * HEAD_DIM)
                : nullptr;
            // Vectorized 16-byte load path: each lane grabs 8 halves with one
            // int4 instruction, replacing 8 strided per-half loads. Both K row
            // and shmem destination are 16-byte aligned for HEAD_DIM ∈ {128,
            // 256}. Lanes past LANES_PER_ROW are idle for HEAD_DIM=128.
            if constexpr (std::is_same<KVT, __half>::value) {
                constexpr uint32_t HALVES_PER_LD = 8;
                constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;
                if (lane < LANES_PER_ROW) {
                    const uint32_t base_d = lane * HALVES_PER_LD;
                    int4 zero4; zero4.x = zero4.y = zero4.z = zero4.w = 0;
                    if (t_in_range) {
                        *reinterpret_cast<int4 *>(&s_k[shmem_t * HEAD_DIM + base_d]) =
                            *reinterpret_cast<const int4 *>(&k_src[base_d]);
                        *reinterpret_cast<int4 *>(&s_v[shmem_t * HEAD_DIM + base_d]) =
                            *reinterpret_cast<const int4 *>(&v_src[base_d]);
                    } else {
                        *reinterpret_cast<int4 *>(&s_k[shmem_t * HEAD_DIM + base_d]) = zero4;
                        *reinterpret_cast<int4 *>(&s_v[shmem_t * HEAD_DIM + base_d]) = zero4;
                    }
                }
            } else {
                #pragma unroll
                for (uint32_t r = 0; r < D_PER_LANE; ++r) {
                    const uint32_t d = r * WARP_SIZE + lane;
                    if (t_in_range) {
                        s_k[shmem_t * HEAD_DIM + d] = k_src[d];
                        s_v[shmem_t * HEAD_DIM + d] = v_src[d];
                    } else {
                        s_k[shmem_t * HEAD_DIM + d] = 0.0f;
                        s_v[shmem_t * HEAD_DIM + d] = 0.0f;
                    }
                }
            }
        }
        __syncthreads();

        // Online softmax over BC tokens, score-by-score: each iteration is
        // warp-uniform (all lanes see the same `score`), so the rescale
        // branch never diverges.
        if (warp_active) {
            #pragma unroll 1
            for (uint32_t k = 0; k < BC; ++k) {
                const uint32_t t = t0 + k;
                const bool valid = (t < my_max_kv);

                // Q · K[t]: each lane computes its slice of the dot product,
                // then a warp reduce.
                float local = 0.0f;
                #pragma unroll
                for (uint32_t r = 0; r < D_PER_LANE; ++r) {
                    const float kv = kv_load<KVT>(&s_k[k * HEAD_DIM + r * WARP_SIZE + lane]);
                    local += q_reg[r] * kv;
                }
                const float dot = cuda_helpers::warp_reduce_sum<WARP_SIZE>(local);
                const float score = valid ? (dot * scale) : -INFINITY;

                // Online softmax update.
                const float new_max = fmaxf(running_max, score);
                const float prev_scale = (running_max == -INFINITY)
                    ? 0.0f
                    : __expf(running_max - new_max);
                const float w_k = (score == -INFINITY) ? 0.0f
                                                       : __expf(score - new_max);

                running_sum = running_sum * prev_scale + w_k;
                #pragma unroll
                for (uint32_t r = 0; r < D_PER_LANE; ++r) {
                    const float v = kv_load<KVT>(&s_v[k * HEAD_DIM + r * WARP_SIZE + lane]);
                    VKQ[r] = VKQ[r] * prev_scale + w_k * v;
                }
                running_max = new_max;
            }
        }
        __syncthreads();    // shmem reuse next tile
    }

    // Final write: each lane owns r-th D-shard of its warp's query.
    if (warp_active) {
        const float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
        float * out_ptr = out + static_cast<uint64_t>(q_idx) * out_batch_stride
                              + head * HEAD_DIM;
        #pragma unroll
        for (uint32_t r = 0; r < D_PER_LANE; ++r) {
            out_ptr[r * WARP_SIZE + lane] = VKQ[r] * inv_sum;
        }
    }
}

static int prefill_attn_br() {
    static const int v = []() {
        const char *e = std::getenv("QW3_PREFILL_ATTN_BR");
        if (!e) return 16;
        int n = std::atoi(e);
        if (n == 4 || n == 8 || n == 16 || n == 32) return n;
        return 16;
    }();
    return v;
}

template <typename KVT>
static bool dispatch_launch_prefill(
        float * out,
        const float * q,
        uint32_t q_stride,
        const KVT * k_cache,
        const KVT * v_cache,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t batch,
        uint32_t base_seq_len,
        uint32_t q_batch_stride,
        uint32_t out_batch_stride,
        float    scale,
        cudaStream_t stream) {
    if (head_dim != 128 && head_dim != 256) return false;
    if (batch == 0) return true;

    constexpr uint32_t BC = 32;
    const int br = prefill_attn_br();

    auto launch = [&](auto BR_v) {
        constexpr uint32_t BR = decltype(BR_v)::value;
        const uint32_t n_blocks_q = (batch + BR - 1) / BR;
        const dim3 grid(n_heads, n_blocks_q, 1);
        const dim3 block(BR * 32);
        if (head_dim == 128) {
            fattn_prefill_kernel<128, KVT, BR, BC><<<grid, block, 0, stream>>>(
                out, q, q_stride, k_cache, v_cache,
                n_heads, n_kv_heads, base_seq_len, batch,
                q_batch_stride, out_batch_stride, scale);
        } else {
            fattn_prefill_kernel<256, KVT, BR, BC><<<grid, block, 0, stream>>>(
                out, q, q_stride, k_cache, v_cache,
                n_heads, n_kv_heads, base_seq_len, batch,
                q_batch_stride, out_batch_stride, scale);
        }
    };
    if (br == 32)      launch(std::integral_constant<uint32_t, 32>{});
    else if (br == 16) launch(std::integral_constant<uint32_t, 16>{});
    else if (br == 8)  launch(std::integral_constant<uint32_t, 8>{});
    else               launch(std::integral_constant<uint32_t, 4>{});
    return true;
}

bool launch_fattn_prefill_f16(
        float *       out,
        const float * q,
        uint32_t      q_stride,
        const void  * k_cache,
        const void  * v_cache,
        uint32_t      n_heads,
        uint32_t      n_kv_heads,
        uint32_t      head_dim,
        uint32_t      batch,
        uint32_t      base_seq_len,
        uint32_t      q_batch_stride,
        uint32_t      out_batch_stride,
        float         scale,
        cudaStream_t  stream) {
    return dispatch_launch_prefill<__half>(
        out, q, q_stride,
        static_cast<const __half *>(k_cache),
        static_cast<const __half *>(v_cache),
        n_heads, n_kv_heads, head_dim, batch, base_seq_len,
        q_batch_stride, out_batch_stride, scale, stream);
}

} // namespace ported
} // namespace qw3
