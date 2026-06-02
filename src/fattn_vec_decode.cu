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
//   - Quantized K / V caches (qw3 stores K/V as FP16 by default, FP32 opt-in via QW3_KV_DTYPE=fp32).
//   - Multi-column / batched-Q (decode is always 1 token here).
//   - ALiBi slopes, logit-softcap, attention sinks.
//   - Mask tensor (causal is implicit: t < seq_len).
//   - F16/BF16 dispatch hairball — KVT templated on float|__half only, both supported at runtime.
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
#include <cstring>
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
    for (uint32_t s = 0; s < NSPLIT; ++s) {
        if (s_sum[s] > 0.0f && s_max[s] > gmax) gmax = s_max[s];
    }

    // Now combine numerator and denominator:
    // gsum = Σ_s sum_s * exp(max_s - gmax)
    // gnum[d] = Σ_s VKQ_s[d] * exp(max_s - gmax)
    float gsum  = 0.0f;
    float gnum  = 0.0f;
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
// gets large.
//
// Previously this returned 1 for seq_len <= 1024, which left Blackwell with
// only 24 blocks (one per Q-head) × 4 warps = 96 warps total — under 5% of
// the 128 SMs' warp capacity. Each block then sequentially walked the full
// KV tail at ~3 GB/s effective, taking ~315 µs/call (vs the ~2 µs the read
// is actually worth). The kernel is latency- / sync-bound, not bandwidth-
// bound, so adding more parallel splits is nearly free in HBM traffic and
// linear in throughput.
//
// New policy: aim for per_split ≈ 128 KV tokens. That keeps each block's
// inner loop at ~32 iterations (one warp wave per ~10 tokens), enough work
// to amortize the combine-kernel launch but small enough that 24 heads × N
// splits fills the SMs.
//
//   seq_len <= 64   : 1
//   seq_len <= 128  : 2     (256 KV tokens / per_split=128 per block)
//   seq_len <= 256  : 4
//   seq_len <= 512  : 4
//   else            : 8     (max NSPLIT instantiated)
//
// Override via QW3_FATTN_NSPLIT=N (1/2/4/8) for parity diffs.
static inline uint32_t pick_nsplit(uint32_t seq_len) {
    static const uint32_t kForced = []() -> uint32_t {
        const char *e = std::getenv("QW3_FATTN_NSPLIT");
        if (!e) return 0;
        const int v = std::atoi(e);
        if (v == 1 || v == 2 || v == 4 || v == 8 || v == 16 || v == 32 || v == 64) {
            return static_cast<uint32_t>(v);
        }
        return 0;
    }();
    if (kForced) return kForced;
    if (seq_len <= 64)    return 1;
    if (seq_len <= 128)   return 2;
    if (seq_len <= 512)   return 4;
    if (seq_len <= 2048)  return 8;
    if (seq_len <= 8192)  return 16;
    if (seq_len <= 32768) return 32;
    return 64;
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
    DISPATCH(128, 1) DISPATCH(128, 2) DISPATCH(128, 4) DISPATCH(128, 8) DISPATCH(128, 16)
    DISPATCH(128, 32) DISPATCH(128, 64)
    DISPATCH(256, 1) DISPATCH(256, 2) DISPATCH(256, 4) DISPATCH(256, 8) DISPATCH(256, 16)
    DISPATCH(256, 32) DISPATCH(256, 64)
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

// ============================================================================
// Tensor-core (MMA m16n8k16 fp16) prefill flash-attention kernel.
//
// Same algorithmic shape as fattn_prefill_kernel but with the QK^T and PV
// inner loops replaced by MMA tensor-core fragments. Online softmax is row-
// fused so the [BR, BC] score tile lives only in shmem, never HBM.
//
// Geometry (HEAD_DIM=256):
//   - Block: 4 warps x 32 = 128 threads.
//   - BR=16 queries per block (one m16 row per QK^T MMA).
//   - BC=32 KV tokens per tile.
//   - QK^T: each of 4 warps owns an 8-token slice of BC; QK^T fragment is
//     m16n8k16 (1 m-tile, HEAD_DIM/16=16 k-steps, 1 n-tile per warp).
//     Each warp produces 4 floats of S per lane covering its 16x8 slice.
//     S is staged through shmem [BR, BC] FP32 so all warps see the full row.
//   - Softmax: one warp owns 4 rows; each lane scans BC scores from shmem,
//     reduces row max/sum, updates online (m_i, l_i, alpha_i), and writes P
//     back to shmem as FP16.
//   - PV: each warp owns 64 of HEAD_DIM dims (HEAD_DIM/4 = 64). PV fragment
//     is m16n8k16 (1 m-tile, BC/16=2 k-steps, 8 n-tiles per warp).
//     Output accumulators are FP32 and stay in registers across tiles.
//
// Shmem (HEAD_DIM=256, all FP16 except S/m/l/alpha):
//   Q   [16][256]     8 KB
//   K   [32][256]    16 KB
//   V_T [256][32]    16 KB        (V transposed at load: PV B-frag = contig)
//   S   [16][32]      2 KB FP32
//   P   [16][32]      1 KB FP16
//   m,l,alpha[16]   192 B FP32
//   ----                  -------
//                       ~43 KB    (under 48 KB static-shmem cap on sm_80+)
//
// Causal mask: applied per (q_idx, kv_idx) at the score level, identical to
// the SIMT kernel.
//
// V is loaded into shmem in transposed [HEAD_DIM][BC] order so the PV
// B-fragment loads are 32-bit reads of consecutive halves down the K (=BC)
// dimension. Trade: scattered shmem writes during the V load (8 strided
// stores per thread) instead of vectorized writes; PV inner loop benefits
// from this for every K-step on every tile.
// ============================================================================

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define QW3_MMA_AVAILABLE 1
#else
#define QW3_MMA_AVAILABLE 0
#endif

namespace mma_detail {

// PTX wrapper: m16n8k16.row.col.f32.f16.f16.f32. Non-volatile so ptxas can
// schedule MMAs around loads for ILP. Same calling convention as the int8
// MMA in mmq_q8.cu.
__device__ __forceinline__ void mma_m16n8k16_f16f16f32(
        float &c0, float &c1, float &c2, float &c3,
        unsigned a0, unsigned a1, unsigned a2, unsigned a3,
        unsigned b0, unsigned b1) {
#if QW3_MMA_AVAILABLE
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1));
#else
    (void)c0; (void)c1; (void)c2; (void)c3;
    (void)a0; (void)a1; (void)a2; (void)a3;
    (void)b0; (void)b1;
#endif
}

// Pack two consecutive halves at p[0], p[1] into a single .b32 register.
__device__ __forceinline__ unsigned pack2h(const __half *p) {
    __half2 v = *reinterpret_cast<const __half2 *>(p);
    return *reinterpret_cast<unsigned *>(&v);
}

__device__ __forceinline__ unsigned pack2h(__half a, __half b) {
    __half2 v = __halves2half2(a, b);
    return *reinterpret_cast<unsigned *>(&v);
}

// ldmatrix.x4 .b16 — loads a 16x16 fp16 tile from shmem into 4 .b32 regs per
// lane, distributing the result so that each lane holds the A fragment of a
// m16n8k16.row.col MMA (i.e. lane t -> rows {t/4, t/4+8}, cols
// {t%4*2..+1, t%4*2+8..+9}).
//
// Lane addressing: lane provides the shmem address of (row=lane%16,
// col_halves=(lane/16)*8). Stride is in halves between rows. The shmem region
// must be a contiguous 16x16 fp16 tile starting at `base` (i.e.
// base[row*stride_h + col]).
__device__ __forceinline__ void ldmatrix_x4_a(
        unsigned &a0, unsigned &a1, unsigned &a2, unsigned &a3,
        const __half *base, uint32_t stride_h) {
#if QW3_MMA_AVAILABLE
    const uint32_t lane = threadIdx.x % 32;
    const __half *p = base + (lane & 15) * stride_h + ((lane >> 4) * 8);
    unsigned smem = __cvta_generic_to_shared(p);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        : "r"(smem));
#else
    (void)a0; (void)a1; (void)a2; (void)a3; (void)base; (void)stride_h;
#endif
}

// ldmatrix.x2.trans .b16 — loads two 8x8 fp16 tiles from row-major shmem and
// transposes each 8x8 during the load. Result distribution maps directly to
// the B fragment of m16n8k16.row.col (col-major B):
//
//   lane t in [0, 32) holds:
//     b0: (s[k0+2*(t%4)+0, n0+t/4], s[k0+2*(t%4)+1, n0+t/4])  — tile 0
//     b1: (s[k0+2*(t%4)+8, n0+t/4], s[k0+2*(t%4)+9, n0+t/4])  — tile 1
//
// For a row-major source s[t, d] with stride_h halves between K rows, base
// must point to s[k0, n0]. Lanes 0..7 supply tile-0 row addresses (rows
// k0..k0+7 of the column n0..n0+7); lanes 8..15 supply tile-1 row addresses
// (rows k0+8..k0+15). Lanes 16..31 still receive the broadcast result.
__device__ __forceinline__ void ldmatrix_x2_b_trans(
        unsigned &b0, unsigned &b1,
        const __half *base, uint32_t stride_h) {
#if QW3_MMA_AVAILABLE
    const uint32_t lane = threadIdx.x % 32;
    const __half *p = base + (lane & 15) * stride_h;
    unsigned smem = __cvta_generic_to_shared(p);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0,%1}, [%2];"
        : "=r"(b0), "=r"(b1)
        : "r"(smem));
#else
    (void)b0; (void)b1; (void)base; (void)stride_h;
#endif
}

// cp.async.cg.shared.global of 16 bytes from gmem -> smem. Non-blocking;
// wait_group(0) drains all in-flight commits in this thread. Falls back to a
// vectorized synchronous load on pre-Ampere targets.
__device__ __forceinline__ void cp_async_cg_16(
        __half *smem_dst, const __half *gmem_src) {
#if QW3_MMA_AVAILABLE
    unsigned dst = __cvta_generic_to_shared(smem_dst);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst), "l"(gmem_src));
#else
    *reinterpret_cast<int4 *>(smem_dst) = *reinterpret_cast<const int4 *>(gmem_src);
#endif
}

__device__ __forceinline__ void cp_async_zero_16(__half *smem_dst) {
#if QW3_MMA_AVAILABLE
    int4 z; z.x = z.y = z.z = z.w = 0;
    *reinterpret_cast<int4 *>(smem_dst) = z;
#else
    *reinterpret_cast<int4 *>(smem_dst) = make_int4(0, 0, 0, 0);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if QW3_MMA_AVAILABLE
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
#if QW3_MMA_AVAILABLE
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

} // namespace mma_detail

template <uint32_t HEAD_DIM, uint32_t BR, uint32_t BC>
__global__ void
__launch_bounds__(128, 2)
fattn_prefill_mma_kernel(
        float       * __restrict__ out,
        const float * __restrict__ q,
        uint32_t     q_stride,
        const __half* __restrict__ k_cache,
        const __half* __restrict__ v_cache,
        uint32_t     n_heads,
        uint32_t     n_kv_heads,
        uint32_t     base_seq_len,
        uint32_t     batch,
        uint32_t     q_batch_stride,
        uint32_t     out_batch_stride,
        float        scale) {
    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint32_t NWARPS    = 4;
    constexpr uint32_t MMA_M     = 16;
    constexpr uint32_t MMA_N     = 8;
    constexpr uint32_t MMA_K     = 16;
    static_assert(BR == 16, "MMA prefill kernel requires BR=16");
    static_assert(BC == 32, "MMA prefill kernel requires BC=32");
    static_assert(HEAD_DIM == 256 || HEAD_DIM == 128,
                  "MMA prefill kernel supports HEAD_DIM 128 or 256");
    static_assert(HEAD_DIM % MMA_K == 0, "HEAD_DIM must be a multiple of MMA_K");

    constexpr uint32_t QK_KSTEPS  = HEAD_DIM / MMA_K;        // 16 (or 8)
    constexpr uint32_t QK_N_PER_WARP = BC / NWARPS;          // 8 (= MMA_N)
    static_assert(QK_N_PER_WARP == MMA_N, "BC/NWARPS must equal MMA_N");
    constexpr uint32_t PV_KSTEPS  = BC / MMA_K;              // 2
    constexpr uint32_t HD_PER_WARP = HEAD_DIM / NWARPS;      // 64 (or 32)
    constexpr uint32_t PV_N_TILES  = HD_PER_WARP / MMA_N;    // 8 (or 4)

    const uint32_t head    = blockIdx.x;
    const uint32_t block_q = blockIdx.y;
    const uint32_t tid     = threadIdx.x;
    const uint32_t warp    = tid / WARP_SIZE;
    const uint32_t lane    = tid % WARP_SIZE;
    if (head >= n_heads) return;

    const uint32_t kv_head = head / (n_heads / n_kv_heads);

    // MMA fragment lane layout (m16n8k16.row.col):
    //   group_id = lane/4 picks row pair {group_id, group_id+8}
    //   in_group = lane%4 picks the 2-col pair within K (or N for B).
    const uint32_t group_id = lane / 4;
    const uint32_t in_group = lane % 4;

    uint32_t block_max_q = block_q * BR + BR;
    if (block_max_q > batch) block_max_q = batch;
    const uint32_t block_max_kv = base_seq_len + block_max_q;   // exclusive

    // ---- Shared memory layout -------------------------------------------
    extern __shared__ char smem_raw[];
    char *p = smem_raw;
    __half *s_Q   = reinterpret_cast<__half *>(p); p += BR       * HEAD_DIM * sizeof(__half);
    __half *s_K   = reinterpret_cast<__half *>(p); p += BC       * HEAD_DIM * sizeof(__half);
    __half *s_VT  = reinterpret_cast<__half *>(p); p += HEAD_DIM * BC       * sizeof(__half);
    float  *s_S   = reinterpret_cast<float  *>(p); p += BR       * BC       * sizeof(float);
    __half *s_P   = reinterpret_cast<__half *>(p); p += BR       * BC       * sizeof(__half);
    float  *s_m   = reinterpret_cast<float  *>(p); p += BR                  * sizeof(float);
    float  *s_l   = reinterpret_cast<float  *>(p); p += BR                  * sizeof(float);
    float  *s_alpha = reinterpret_cast<float *>(p); p += BR                 * sizeof(float);

    // ---- Load Q into shmem (FP32 -> FP16) -------------------------------
    // BR*HEAD_DIM = 16*256 = 4096 elements. 128 threads -> 32 elts/thread.
    // Lane reads HEAD_DIM/WARP_SIZE elements per query row; queries split
    // by warp_id (4 warps x 4 rows = 16 = BR).
    {
        constexpr uint32_t ROWS_PER_WARP = BR / NWARPS;          // 4
        for (uint32_t r = 0; r < ROWS_PER_WARP; ++r) {
            const uint32_t q_row = warp * ROWS_PER_WARP + r;
            const uint32_t q_idx = block_q * BR + q_row;
            const bool active   = (q_idx < batch);
            const float * q_ptr = active
                ? (q + static_cast<uint64_t>(q_idx) * q_batch_stride
                     + head * q_stride)
                : nullptr;
            #pragma unroll
            for (uint32_t d0 = 0; d0 < HEAD_DIM; d0 += WARP_SIZE) {
                const uint32_t d = d0 + lane;
                const float v = active ? q_ptr[d] : 0.0f;
                s_Q[q_row * HEAD_DIM + d] = __float2half(v);
            }
        }
    }

    // ---- Initialize per-row softmax state --------------------------------
    if (tid < BR) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }

    // ---- Output accumulator (FP32 in registers) --------------------------
    // Each warp owns HD_PER_WARP=64 (or 32 at HD=128) dims of the head.
    // PV fragment lays out 4 floats per thread per (m=16, n=8) tile, so each
    // thread holds PV_N_TILES * 4 floats.
    float O_acc[PV_N_TILES][4];
    #pragma unroll
    for (uint32_t n = 0; n < PV_N_TILES; ++n) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) O_acc[n][i] = 0.0f;
    }

    __syncthreads();

    // =====================================================================
    // Tile loop over KV.
    // =====================================================================
    for (uint32_t t0 = 0; t0 < block_max_kv; t0 += BC) {

        // ---- Load K tile (row-major) into s_K -------------------------
        // 32 tokens x 256 halves = 8192 halves; int4 loads of 8 halves =>
        // 1024 vectors / 128 threads = 8 vectors per thread.
        {
            constexpr uint32_t HALVES_PER_LD = 8;
            constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;   // 32 (or 16)
            constexpr uint32_t TOKENS_PER_PASS = 128 / LANES_PER_ROW;     // 4 (or 8)
            constexpr uint32_t N_PASS = BC / TOKENS_PER_PASS;             // 8 (or 4)
            const uint32_t row_in_pass = tid / LANES_PER_ROW;
            const uint32_t col_lane    = tid % LANES_PER_ROW;
            #pragma unroll
            for (uint32_t pass = 0; pass < N_PASS; ++pass) {
                const uint32_t shm_t = pass * TOKENS_PER_PASS + row_in_pass;
                const uint32_t t     = t0 + shm_t;
                const uint32_t base_d = col_lane * HALVES_PER_LD;
                int4 *dst = reinterpret_cast<int4 *>(&s_K[shm_t * HEAD_DIM + base_d]);
                if (t < block_max_kv) {
                    const __half *k_src = k_cache
                        + static_cast<uint64_t>(t) * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    *dst = *reinterpret_cast<const int4 *>(k_src);
                } else {
                    int4 z; z.x = z.y = z.z = z.w = 0;
                    *dst = z;
                }
            }
        }

        // ---- Load V tile transposed: s_VT[d, t] = V[t, d] -------------
        // Same vectorized HBM read as K (8 halves at a time), but stores to
        // 8 separate shmem locations along the d axis.
        {
            constexpr uint32_t HALVES_PER_LD = 8;
            constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;
            constexpr uint32_t TOKENS_PER_PASS = 128 / LANES_PER_ROW;
            constexpr uint32_t N_PASS = BC / TOKENS_PER_PASS;
            const uint32_t row_in_pass = tid / LANES_PER_ROW;
            const uint32_t col_lane    = tid % LANES_PER_ROW;
            #pragma unroll
            for (uint32_t pass = 0; pass < N_PASS; ++pass) {
                const uint32_t shm_t = pass * TOKENS_PER_PASS + row_in_pass;
                const uint32_t t     = t0 + shm_t;
                const uint32_t base_d = col_lane * HALVES_PER_LD;
                __half v8[HALVES_PER_LD];
                if (t < block_max_kv) {
                    const __half *v_src = v_cache
                        + static_cast<uint64_t>(t) * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    *reinterpret_cast<int4 *>(v8) = *reinterpret_cast<const int4 *>(v_src);
                } else {
                    *reinterpret_cast<int4 *>(v8) = make_int4(0,0,0,0);
                }
                #pragma unroll
                for (uint32_t i = 0; i < HALVES_PER_LD; ++i) {
                    s_VT[(base_d + i) * BC + shm_t] = v8[i];
                }
            }
        }

        __syncthreads();

        // ================================================================
        // QK^T MMA. Each warp computes its 16x8 slice of S = Q * K^T.
        // ================================================================
        // Lane layout for MMA m16n8k16.row.col.f16:
        //   A (Q)  row-major:  thread t holds rows {t/4, t/4+8} x cols
        //                       {t%4*2, t%4*2+1, t%4*2+8, t%4*2+9}
        //                       packed into 4 .b32 regs (a0..a3).
        //   B (K)  col-major:  thread t holds rows {t%4*2, t%4*2+1, t%4*2+8,
        //                       t%4*2+9} x cols {t/4} packed into 2 .b32
        //                       regs (b0..b1).
        //   C (S)  row-major:  thread t holds rows {t/4, t/4+8} x cols
        //                       {t%4*2, t%4*2+1} as 4 floats c0..c3.
        // For QK^T, "rows of B" along K dim = HEAD_DIM elements (col-major
        // means stride along K is 1). K is stored in shmem as s_K[t][d];
        // viewed as col-major [d, t] this is a stride-256 along d, which
        // is exactly what the row.col MMA wants when we set ldb = HEAD_DIM
        // and treat B as col-major over (d, n).
        float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f;
        #pragma unroll
        for (uint32_t ks = 0; ks < QK_KSTEPS; ++ks) {
            const uint32_t k0 = ks * MMA_K;
            // A (Q) fragment from s_Q via ldmatrix.x4 — single PTX, all 4 a regs.
            unsigned a0, a1, a2, a3;
            mma_detail::ldmatrix_x4_a(a0, a1, a2, a3,
                                      &s_Q[0 * HEAD_DIM + k0], HEAD_DIM);
            // B (K) fragment. B is col-major; rows of B = K dim (= HEAD_DIM).
            // For warp `warp`, its 8 N-cols start at warp * MMA_N. The MMA
            // expects, per thread, two pairs along the K dim at column = group_id.
            const uint32_t kv_col = warp * MMA_N + group_id;
            const __half *krow0 = &s_K[kv_col * HEAD_DIM + k0 + in_group * 2];
            unsigned b0 = mma_detail::pack2h(krow0);              // k[col, k0+2c+0..1]
            unsigned b1 = mma_detail::pack2h(krow0 + 8);          // k[col, k0+2c+8..9]
            mma_detail::mma_m16n8k16_f16f16f32(c00, c01, c02, c03,
                                               a0, a1, a2, a3, b0, b1);
        }

        // Write S slice to shmem with causal mask. Each thread holds 4
        // values: (row=group_id,    col=warp*8 + in_group*2 + 0)
        //         (row=group_id,    col=warp*8 + in_group*2 + 1)
        //         (row=group_id+8,  col=warp*8 + in_group*2 + 0)
        //         (row=group_id+8,  col=warp*8 + in_group*2 + 1)
        {
            const uint32_t base_col = warp * MMA_N + in_group * 2;
            const uint32_t row_a = group_id;
            const uint32_t row_b = group_id + 8;
            const uint32_t q_idx_a = block_q * BR + row_a;
            const uint32_t q_idx_b = block_q * BR + row_b;
            const uint32_t my_max_kv_a = base_seq_len + q_idx_a + 1;
            const uint32_t my_max_kv_b = base_seq_len + q_idx_b + 1;
            const bool a_active = (q_idx_a < batch);
            const bool b_active = (q_idx_b < batch);
            const uint32_t t_a0 = t0 + base_col;
            const uint32_t t_a1 = t0 + base_col + 1;
            float v00 = (a_active && t_a0 < my_max_kv_a) ? c00 * scale : -INFINITY;
            float v01 = (a_active && t_a1 < my_max_kv_a) ? c01 * scale : -INFINITY;
            float v02 = (b_active && t_a0 < my_max_kv_b) ? c02 * scale : -INFINITY;
            float v03 = (b_active && t_a1 < my_max_kv_b) ? c03 * scale : -INFINITY;
            s_S[row_a * BC + base_col + 0] = v00;
            s_S[row_a * BC + base_col + 1] = v01;
            s_S[row_b * BC + base_col + 0] = v02;
            s_S[row_b * BC + base_col + 1] = v03;
        }

        __syncthreads();

        // ================================================================
        // Online softmax. One warp owns 4 rows; lane scans BC scores.
        // ================================================================
        // 4 warps x 4 rows = 16 = BR. Each warp sweeps its 4 rows; for each
        // row, all 32 lanes load BC=32 scores (lane l reads s_S[row*BC + l]).
        {
            constexpr uint32_t ROWS_PER_WARP = BR / NWARPS;          // 4
            #pragma unroll
            for (uint32_t r = 0; r < ROWS_PER_WARP; ++r) {
                const uint32_t row = warp * ROWS_PER_WARP + r;
                const float s = s_S[row * BC + lane];                 // lane = col
                const float row_max_local = s;
                const float row_max = cuda_helpers::warp_reduce_max<32>(row_max_local);
                const float prev_m = s_m[row];
                const float new_m  = fmaxf(prev_m, row_max);
                const float p_val  = (s == -INFINITY) ? 0.0f
                                                      : __expf(s - new_m);
                const float row_p_sum = cuda_helpers::warp_reduce_sum<32>(p_val);
                const float prev_l   = s_l[row];
                const float alpha    = (prev_m == -INFINITY) ? 0.0f
                                                             : __expf(prev_m - new_m);
                const float new_l    = prev_l * alpha + row_p_sum;
                if (lane == 0) {
                    s_m[row]     = new_m;
                    s_l[row]     = new_l;
                    s_alpha[row] = alpha;
                }
                // Write P (FP16) — lane l writes col l for this row.
                s_P[row * BC + lane] = __float2half(p_val);
            }
        }

        __syncthreads();

        // ================================================================
        // Rescale O_acc by alpha_row before adding new PV contribution.
        // ================================================================
        // Each thread holds 4 floats per N-tile, mapping to rows
        //   [group_id, group_id+8] x cols [in_group*2, in_group*2+1].
        // The alpha factor depends only on the row.
        const float alpha_a = s_alpha[group_id];
        const float alpha_b = s_alpha[group_id + 8];
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            O_acc[n][0] *= alpha_a;
            O_acc[n][1] *= alpha_a;
            O_acc[n][2] *= alpha_b;
            O_acc[n][3] *= alpha_b;
        }

        // ================================================================
        // PV MMA. O += P @ V. P[BR, BC] @ V[BC, HEAD_DIM] -> O[BR, HEAD_DIM].
        // For MMA m16n8k16.row.col.f16:
        //   A (P) row-major: rows = BR, K = BC. PV_KSTEPS = BC/MMA_K = 2.
        //   B (V) col-major along K: viewing V as [HEAD_DIM, BC] col-major
        //         means stride along K is 1, stride along N is BC. Our s_VT
        //         is exactly [d=HEAD_DIM, t=BC] row-major in shmem, which
        //         equals col-major when we re-label (d -> N axis, t -> K
        //         axis). So ldb = BC.
        // Each warp owns N range [warp*HD_PER_WARP, (warp+1)*HD_PER_WARP) of
        // the head dim, split into PV_N_TILES tiles of MMA_N=8.
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            const uint32_t n0 = warp * HD_PER_WARP + n * MMA_N;
            #pragma unroll
            for (uint32_t ks = 0; ks < PV_KSTEPS; ++ks) {
                const uint32_t k0 = ks * MMA_K;
                // A (P) fragment from s_P via ldmatrix.x4. ldb = BC.
                unsigned a0, a1, a2, a3;
                mma_detail::ldmatrix_x4_a(a0, a1, a2, a3,
                                          &s_P[0 * BC + k0], BC);
                // B (V) fragment. col_in_warp_n = group_id selects an N col
                // within the 8-col warp slice. Two pairs along K = BC.
                const uint32_t out_col = n0 + group_id;
                const __half *vrow0 = &s_VT[out_col * BC + k0 + in_group * 2];
                unsigned b0 = mma_detail::pack2h(vrow0);
                unsigned b1 = mma_detail::pack2h(vrow0 + 8);
                mma_detail::mma_m16n8k16_f16f16f32(
                    O_acc[n][0], O_acc[n][1], O_acc[n][2], O_acc[n][3],
                    a0, a1, a2, a3, b0, b1);
            }
        }

        __syncthreads();   // s_K, s_VT, s_S, s_P reused next tile
    }

    // =====================================================================
    // Final write. Each thread holds 4 FP32 outputs per N-tile:
    //   (row=group_id   , col=n0 + in_group*2 + 0)
    //   (row=group_id   , col=n0 + in_group*2 + 1)
    //   (row=group_id+8 , col=n0 + in_group*2 + 0)
    //   (row=group_id+8 , col=n0 + in_group*2 + 1)
    // Divide by l_i and store to global out.
    // =====================================================================
    const float l_a = s_l[group_id];
    const float l_b = s_l[group_id + 8];
    const float inv_la = (l_a > 0.0f) ? (1.0f / l_a) : 0.0f;
    const float inv_lb = (l_b > 0.0f) ? (1.0f / l_b) : 0.0f;
    const uint32_t q_idx_a = block_q * BR + group_id;
    const uint32_t q_idx_b = block_q * BR + group_id + 8;
    float * const out_a = (q_idx_a < batch)
        ? (out + static_cast<uint64_t>(q_idx_a) * out_batch_stride + head * HEAD_DIM)
        : nullptr;
    float * const out_b = (q_idx_b < batch)
        ? (out + static_cast<uint64_t>(q_idx_b) * out_batch_stride + head * HEAD_DIM)
        : nullptr;
    #pragma unroll
    for (uint32_t n = 0; n < PV_N_TILES; ++n) {
        const uint32_t n0 = warp * HD_PER_WARP + n * MMA_N;
        const uint32_t col = n0 + in_group * 2;
        if (out_a) {
            out_a[col + 0] = O_acc[n][0] * inv_la;
            out_a[col + 1] = O_acc[n][1] * inv_la;
        }
        if (out_b) {
            out_b[col + 0] = O_acc[n][2] * inv_lb;
            out_b[col + 1] = O_acc[n][3] * inv_lb;
        }
    }
}

// ============================================================================
// Piped variant: 2-stage cp.async on K loads + Q kept in registers across
// the entire K-loop (single ldmatrix.x4 per ks before the loop). V load stays
// synchronous (the strided transposed write doesn't fit cp.async; switching
// V to row-major + ldmatrix.trans is a follow-up).
//
// Shmem grows by BC*HEAD_DIM*sizeof(half) for the second K buffer:
//   HD=256: 43 -> 59 KB; HD=128: 24 -> 32 KB. Both under 99 KB cap.
// Q[16*256] is dropped since Q is in registers (saves 8 KB at HD=256).
//
// Net per-tile change vs the synchronous kernel:
//   - K HBM read overlaps the previous tile's QK+softmax+PV (~3-4 us hidden)
//   - Q ldmatrix.x4 issued QK_KSTEPS times instead of QK_KSTEPS*N_TILES_KV
// ============================================================================
template <uint32_t HEAD_DIM, uint32_t BR, uint32_t BC>
__global__ void
__launch_bounds__(128, 2)
fattn_prefill_mma_kernel_pipe(
        float       * __restrict__ out,
        const float * __restrict__ q,
        uint32_t     q_stride,
        const __half* __restrict__ k_cache,
        const __half* __restrict__ v_cache,
        uint32_t     n_heads,
        uint32_t     n_kv_heads,
        uint32_t     base_seq_len,
        uint32_t     batch,
        uint32_t     q_batch_stride,
        uint32_t     out_batch_stride,
        float        scale) {
    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint32_t NWARPS    = 4;
    constexpr uint32_t MMA_M     = 16;
    constexpr uint32_t MMA_N     = 8;
    constexpr uint32_t MMA_K     = 16;
    static_assert(BR == 16, "MMA prefill kernel requires BR=16");
    static_assert(BC == 32, "MMA prefill kernel requires BC=32");
    static_assert(HEAD_DIM == 256 || HEAD_DIM == 128,
                  "MMA prefill kernel supports HEAD_DIM 128 or 256");
    static_assert(HEAD_DIM % MMA_K == 0, "HEAD_DIM must be a multiple of MMA_K");

    constexpr uint32_t QK_KSTEPS  = HEAD_DIM / MMA_K;
    constexpr uint32_t QK_N_PER_WARP = BC / NWARPS;
    static_assert(QK_N_PER_WARP == MMA_N, "BC/NWARPS must equal MMA_N");
    constexpr uint32_t PV_KSTEPS  = BC / MMA_K;
    constexpr uint32_t HD_PER_WARP = HEAD_DIM / NWARPS;
    constexpr uint32_t PV_N_TILES  = HD_PER_WARP / MMA_N;

    const uint32_t head    = blockIdx.x;
    const uint32_t block_q = blockIdx.y;
    const uint32_t tid     = threadIdx.x;
    const uint32_t warp    = tid / WARP_SIZE;
    const uint32_t lane    = tid % WARP_SIZE;
    if (head >= n_heads) return;
    const uint32_t kv_head = head / (n_heads / n_kv_heads);
    const uint32_t group_id = lane / 4;
    const uint32_t in_group = lane % 4;

    uint32_t block_max_q = block_q * BR + BR;
    if (block_max_q > batch) block_max_q = batch;
    const uint32_t block_max_kv = base_seq_len + block_max_q;

    // ---- Shared memory layout (no s_Q: Q is in registers) -------------------
    // Two K buffers for cp.async ping-pong; single V_T buffer (V load is sync).
    extern __shared__ char smem_raw[];
    char *p = smem_raw;
    __half *s_K[2];
    s_K[0] = reinterpret_cast<__half *>(p); p += BC       * HEAD_DIM * sizeof(__half);
    s_K[1] = reinterpret_cast<__half *>(p); p += BC       * HEAD_DIM * sizeof(__half);
    __half *s_VT  = reinterpret_cast<__half *>(p); p += HEAD_DIM * BC * sizeof(__half);
    float  *s_S   = reinterpret_cast<float  *>(p); p += BR       * BC * sizeof(float);
    __half *s_P   = reinterpret_cast<__half *>(p); p += BR       * BC * sizeof(__half);
    float  *s_m   = reinterpret_cast<float  *>(p); p += BR            * sizeof(float);
    float  *s_l   = reinterpret_cast<float  *>(p); p += BR            * sizeof(float);
    float  *s_alpha = reinterpret_cast<float *>(p); p += BR           * sizeof(float);

    // ---- Load Q into registers via ldmatrix ----------------------------------
    // Stage Q to a tiny shmem scratchpad (BR*HEAD_DIM halves = 16*256 = 8 KB),
    // do the ldmatrix, then drop. We reuse the s_K[0] buffer for this — it'll
    // be overwritten by the first K-tile load before any QK consumption.
    __half *s_Q_scratch = s_K[0];
    {
        constexpr uint32_t ROWS_PER_WARP = BR / NWARPS;          // 4
        for (uint32_t r = 0; r < ROWS_PER_WARP; ++r) {
            const uint32_t q_row = warp * ROWS_PER_WARP + r;
            const uint32_t q_idx = block_q * BR + q_row;
            const bool active   = (q_idx < batch);
            const float * q_ptr = active
                ? (q + static_cast<uint64_t>(q_idx) * q_batch_stride
                     + head * q_stride)
                : nullptr;
            #pragma unroll
            for (uint32_t d0 = 0; d0 < HEAD_DIM; d0 += WARP_SIZE) {
                const uint32_t d = d0 + lane;
                const float v = active ? q_ptr[d] : 0.0f;
                s_Q_scratch[q_row * HEAD_DIM + d] = __float2half(v);
            }
        }
    }
    __syncthreads();

    // Pre-load all QK_KSTEPS Q fragments into registers via ldmatrix.x4.
    unsigned Q_frag[QK_KSTEPS][4];
    #pragma unroll
    for (uint32_t ks = 0; ks < QK_KSTEPS; ++ks) {
        const uint32_t k0 = ks * MMA_K;
        mma_detail::ldmatrix_x4_a(Q_frag[ks][0], Q_frag[ks][1],
                                  Q_frag[ks][2], Q_frag[ks][3],
                                  &s_Q_scratch[0 * HEAD_DIM + k0], HEAD_DIM);
    }
    __syncthreads();   // s_Q_scratch (= s_K[0]) is now free for K-load reuse.

    if (tid < BR) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }
    float O_acc[PV_N_TILES][4];
    #pragma unroll
    for (uint32_t n = 0; n < PV_N_TILES; ++n) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) O_acc[n][i] = 0.0f;
    }

    // K-tile cp.async loader. Issues 8 cp.async.cg.shared.global of 16 B each
    // per thread, into s_K[stage]. Returns without committing — caller calls
    // cp_async_commit() once per stage.
    auto issue_K_load = [&](uint32_t stage, uint32_t t0) {
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;
        constexpr uint32_t TOKENS_PER_PASS = 128 / LANES_PER_ROW;
        constexpr uint32_t N_PASS = BC / TOKENS_PER_PASS;
        const uint32_t row_in_pass = tid / LANES_PER_ROW;
        const uint32_t col_lane    = tid % LANES_PER_ROW;
        #pragma unroll
        for (uint32_t pass = 0; pass < N_PASS; ++pass) {
            const uint32_t shm_t = pass * TOKENS_PER_PASS + row_in_pass;
            const uint32_t t     = t0 + shm_t;
            const uint32_t base_d = col_lane * HALVES_PER_LD;
            __half *dst = &s_K[stage][shm_t * HEAD_DIM + base_d];
            if (t < block_max_kv) {
                const __half *k_src = k_cache
                    + static_cast<uint64_t>(t) * n_kv_heads * HEAD_DIM
                    + kv_head * HEAD_DIM + base_d;
                mma_detail::cp_async_cg_16(dst, k_src);
            } else {
                mma_detail::cp_async_zero_16(dst);
            }
        }
    };

    // V-tile sync transposed load. Same as the original kernel.
    auto load_V_sync = [&](uint32_t t0) {
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;
        constexpr uint32_t TOKENS_PER_PASS = 128 / LANES_PER_ROW;
        constexpr uint32_t N_PASS = BC / TOKENS_PER_PASS;
        const uint32_t row_in_pass = tid / LANES_PER_ROW;
        const uint32_t col_lane    = tid % LANES_PER_ROW;
        #pragma unroll
        for (uint32_t pass = 0; pass < N_PASS; ++pass) {
            const uint32_t shm_t = pass * TOKENS_PER_PASS + row_in_pass;
            const uint32_t t     = t0 + shm_t;
            const uint32_t base_d = col_lane * HALVES_PER_LD;
            __half v8[HALVES_PER_LD];
            if (t < block_max_kv) {
                const __half *v_src = v_cache
                    + static_cast<uint64_t>(t) * n_kv_heads * HEAD_DIM
                    + kv_head * HEAD_DIM + base_d;
                *reinterpret_cast<int4 *>(v8) = *reinterpret_cast<const int4 *>(v_src);
            } else {
                *reinterpret_cast<int4 *>(v8) = make_int4(0,0,0,0);
            }
            #pragma unroll
            for (uint32_t i = 0; i < HALVES_PER_LD; ++i) {
                s_VT[(base_d + i) * BC + shm_t] = v8[i];
            }
        }
    };

    // Prologue: issue stage 0's K cp.async, then commit. Don't wait yet.
    issue_K_load(0, /*t0=*/0);
    mma_detail::cp_async_commit();

    // =====================================================================
    // Pipelined tile loop: while consuming stage `cur`'s K, issue stage
    // `cur+1`'s K. V load is sync after the K wait (V is consumed only by
    // PV after softmax, so it doesn't need the same prefetch).
    // =====================================================================
    uint32_t cur_stage = 0;
    for (uint32_t t0 = 0; t0 < block_max_kv; t0 += BC) {
        const uint32_t next_t0 = t0 + BC;
        const bool has_next = (next_t0 < block_max_kv);
        const uint32_t next_stage = cur_stage ^ 1u;

        // Issue stage cur+1's K cp.async BEFORE waiting on cur — this lets
        // the new HBM read overlap the QK+softmax+PV consumption below.
        if (has_next) {
            issue_K_load(next_stage, next_t0);
            mma_detail::cp_async_commit();
            // Wait until cur's load is done (1 still in flight = next).
            mma_detail::cp_async_wait_group<1>();
        } else {
            mma_detail::cp_async_wait_group<0>();
        }
        __syncthreads();

        // V is loaded synchronously here — out of the K critical path.
        load_V_sync(t0);

        // ================================================================
        // QK^T MMA. Q is in registers (Q_frag[]); K from s_K[cur_stage].
        // ================================================================
        float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f;
        #pragma unroll
        for (uint32_t ks = 0; ks < QK_KSTEPS; ++ks) {
            const uint32_t k0 = ks * MMA_K;
            const uint32_t kv_col = warp * MMA_N + group_id;
            const __half *krow0 = &s_K[cur_stage][kv_col * HEAD_DIM + k0 + in_group * 2];
            unsigned b0 = mma_detail::pack2h(krow0);
            unsigned b1 = mma_detail::pack2h(krow0 + 8);
            mma_detail::mma_m16n8k16_f16f16f32(
                c00, c01, c02, c03,
                Q_frag[ks][0], Q_frag[ks][1], Q_frag[ks][2], Q_frag[ks][3],
                b0, b1);
        }

        __syncthreads();   // wait for V load (issued above) before PV.

        // Write S slice with causal mask (same as base kernel).
        {
            const uint32_t base_col = warp * MMA_N + in_group * 2;
            const uint32_t row_a = group_id;
            const uint32_t row_b = group_id + 8;
            const uint32_t q_idx_a = block_q * BR + row_a;
            const uint32_t q_idx_b = block_q * BR + row_b;
            const uint32_t my_max_kv_a = base_seq_len + q_idx_a + 1;
            const uint32_t my_max_kv_b = base_seq_len + q_idx_b + 1;
            const bool a_active = (q_idx_a < batch);
            const bool b_active = (q_idx_b < batch);
            const uint32_t t_a0 = t0 + base_col;
            const uint32_t t_a1 = t0 + base_col + 1;
            float v00 = (a_active && t_a0 < my_max_kv_a) ? c00 * scale : -INFINITY;
            float v01 = (a_active && t_a1 < my_max_kv_a) ? c01 * scale : -INFINITY;
            float v02 = (b_active && t_a0 < my_max_kv_b) ? c02 * scale : -INFINITY;
            float v03 = (b_active && t_a1 < my_max_kv_b) ? c03 * scale : -INFINITY;
            s_S[row_a * BC + base_col + 0] = v00;
            s_S[row_a * BC + base_col + 1] = v01;
            s_S[row_b * BC + base_col + 0] = v02;
            s_S[row_b * BC + base_col + 1] = v03;
        }

        __syncthreads();

        // Online softmax (identical to base kernel).
        {
            constexpr uint32_t ROWS_PER_WARP = BR / NWARPS;
            #pragma unroll
            for (uint32_t r = 0; r < ROWS_PER_WARP; ++r) {
                const uint32_t row = warp * ROWS_PER_WARP + r;
                const float s = s_S[row * BC + lane];
                const float row_max_local = s;
                const float row_max = cuda_helpers::warp_reduce_max<32>(row_max_local);
                const float prev_m = s_m[row];
                const float new_m  = fmaxf(prev_m, row_max);
                const float p_val  = (s == -INFINITY) ? 0.0f
                                                      : __expf(s - new_m);
                const float row_p_sum = cuda_helpers::warp_reduce_sum<32>(p_val);
                const float prev_l   = s_l[row];
                const float alpha    = (prev_m == -INFINITY) ? 0.0f
                                                             : __expf(prev_m - new_m);
                const float new_l    = prev_l * alpha + row_p_sum;
                if (lane == 0) {
                    s_m[row]     = new_m;
                    s_l[row]     = new_l;
                    s_alpha[row] = alpha;
                }
                s_P[row * BC + lane] = __float2half(p_val);
            }
        }

        __syncthreads();

        const float alpha_a = s_alpha[group_id];
        const float alpha_b = s_alpha[group_id + 8];
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            O_acc[n][0] *= alpha_a;
            O_acc[n][1] *= alpha_a;
            O_acc[n][2] *= alpha_b;
            O_acc[n][3] *= alpha_b;
        }

        // PV MMA (identical to base kernel).
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            const uint32_t n0 = warp * HD_PER_WARP + n * MMA_N;
            #pragma unroll
            for (uint32_t ks = 0; ks < PV_KSTEPS; ++ks) {
                const uint32_t k0 = ks * MMA_K;
                unsigned a0, a1, a2, a3;
                mma_detail::ldmatrix_x4_a(a0, a1, a2, a3,
                                          &s_P[0 * BC + k0], BC);
                const uint32_t out_col = n0 + group_id;
                const __half *vrow0 = &s_VT[out_col * BC + k0 + in_group * 2];
                unsigned b0 = mma_detail::pack2h(vrow0);
                unsigned b1 = mma_detail::pack2h(vrow0 + 8);
                mma_detail::mma_m16n8k16_f16f16f32(
                    O_acc[n][0], O_acc[n][1], O_acc[n][2], O_acc[n][3],
                    a0, a1, a2, a3, b0, b1);
            }
        }

        cur_stage = next_stage;
        __syncthreads();   // s_VT, s_S, s_P reused next tile
    }

    // Final write (identical to base kernel).
    const float l_a = s_l[group_id];
    const float l_b = s_l[group_id + 8];
    const float inv_la = (l_a > 0.0f) ? (1.0f / l_a) : 0.0f;
    const float inv_lb = (l_b > 0.0f) ? (1.0f / l_b) : 0.0f;
    const uint32_t q_idx_a = block_q * BR + group_id;
    const uint32_t q_idx_b = block_q * BR + group_id + 8;
    float * const out_a = (q_idx_a < batch)
        ? (out + static_cast<uint64_t>(q_idx_a) * out_batch_stride + head * HEAD_DIM)
        : nullptr;
    float * const out_b = (q_idx_b < batch)
        ? (out + static_cast<uint64_t>(q_idx_b) * out_batch_stride + head * HEAD_DIM)
        : nullptr;
    #pragma unroll
    for (uint32_t n = 0; n < PV_N_TILES; ++n) {
        const uint32_t n0 = warp * HD_PER_WARP + n * MMA_N;
        const uint32_t col = n0 + in_group * 2;
        if (out_a) {
            out_a[col + 0] = O_acc[n][0] * inv_la;
            out_a[col + 1] = O_acc[n][1] * inv_la;
        }
        if (out_b) {
            out_b[col + 0] = O_acc[n][2] * inv_lb;
            out_b[col + 1] = O_acc[n][3] * inv_lb;
        }
    }
}

// ============================================================================
// GQA-fused MMA prefill kernel.
//
// Motivation: Qwen 3.6 27B has 24 q-heads / 4 kv-heads (group=6). The plain
// MMA kernel grids on (n_heads, n_blocks_q), so each of the 6 q-heads sharing
// a kv_head launches an independent block and reloads the same K/V from HBM.
// At T=32K the attention tile loop is HBM-bound on K/V reads — fusing the
// 6 q-heads into one block cuts K/V HBM traffic 6×.
//
// Design:
//   Grid: (n_kv_heads, n_blocks_q). Each block handles ALL Q_PER_KV q-heads
//   sharing this kv_head, BR queries per q-head.
//
//   Outer loop: for each tile of BC KV tokens, load K/V from HBM ONCE into
//   shmem (cp.async ping-pong on K, sync on V_T).
//
//   Inner loop: for each q-head h in 0..Q_PER_KV, run QK^T + softmax + PV
//   against the shared K/V tile. Per-head register state:
//     O_acc[Q_PER_KV][PV_N_TILES][4] floats     — output accumulator
//     Q_frag[Q_PER_KV][QK_KSTEPS][4] unsigneds  — Q fragment registers
//   Per-head shmem state:
//     s_m[Q_PER_KV][BR], s_l[Q_PER_KV][BR]      — softmax statistics
//   s_alpha is reused (only live during one head's softmax step).
//
// Register pressure analysis (HD=256, Q_PER_KV=6):
//   O_acc: 6*8*4 = 192 floats  = 768 B/thread
//   Q_frag: 6*16*4 = 384 unsigneds = 1536 B/thread
//   Total fragment regs ~2.3 KB/thread. With launch_bounds(128, 1) each
//   block gets 256 KB / 128 threads = 2 KB/thread → tight; we may spill.
//
// Trade: drop Q_frag from registers and reload it per (tile, q-head) via
// ldmatrix from a small s_Q[Q_PER_KV*BR*HEAD_DIM]=24 KB (HD=256, 6 heads)
// — that costs QK_KSTEPS extra ldmatrix per inner-q-head step but halves
// register pressure. We do this by keeping s_Q persistent across tiles
// (load Q ONCE before the tile loop, reload Q_frag from s_Q per inner step).
//
// Shmem layout (HD=256, Q_PER_KV=6):
//   s_K[2][BC*HEAD_DIM]   = 2*32*256*2  = 32 KB   (cp.async ping-pong)
//   s_VT[HEAD_DIM*BC]     = 256*32*2    = 16 KB
//   s_Q[Q_PER_KV*BR*HD]   = 6*16*256*2  = 48 KB
//   s_S[BR*BC]            = 16*32*4     = 2 KB
//   s_P[BR*BC]            = 16*32*2     = 1 KB
//   s_m,s_l,s_alpha       = ~1 KB
//   Total                 ≈ 100 KB → at the 99 KB cap. Need to drop one
//   K buffer (kill cp.async pipeline) OR reduce s_Q.
//
// Resolution: keep cp.async, drop s_Q to a "load once per outer q-head loop"
// buffer of size BR*HEAD_DIM (8 KB) — load Q for q-head h into s_Q, do ALL
// tiles for that head, swap to next head. But that destroys the "K/V loaded
// once per tile" benefit because V is reused across heads.
//
// Better: keep s_Q sized for ALL Q_PER_KV heads in shmem at once, but drop
// the second K buffer (no cp.async ping-pong — single K buffer + sync wait).
// Net shmem (HD=256, Q_PER_KV=6):
//   s_K[BC*HD]    = 16 KB
//   s_VT[HD*BC]   = 16 KB
//   s_Q[6*BR*HD]  = 48 KB
//   s_S+s_P       = 3 KB
//   misc          = 1 KB
//   Total         = 84 KB ≤ 99 KB cap. ✓
//
// Trade-off: lose cp.async overlap (one HBM stall per tile). But we GAIN
// 6× K/V reuse (only 1 HBM load per 6 head-passes vs 6 HBM loads). For
// long contexts where K/V HBM is the bottleneck, this wins by far.
//
// At HD=128, Q_PER_KV=6:
//   s_K = 8 KB, s_VT = 8 KB, s_Q = 24 KB, s_S+s_P = 3 KB, misc = 1 KB
//   Total = 44 KB → cp.async ping-pong fits (52 KB). We gate on HEAD_DIM.
// ============================================================================

// ============================================================================
// GQA-fused prefill MMA kernel (Q_PER_KV q-heads share one CTA per kv_head).
// See above doc block for invariants. Two cp.async knobs:
//
//   USE_CP_ASYNC_K — issue K[t+1] post-QK so its HBM read overlaps the
//                    current tile's softmax + PV (single-buffered, deferred
//                    wait at the top of the next iter).
//   USE_CP_ASYNC_V — store V row-major (s_V[t, d] instead of the old
//                    transposed s_VT[d, t]) so cp.async's gmem→shmem path
//                    accepts it. V[t+1] is issued post-PV (when s_V is
//                    dead) so its read overlaps the next iter's K wait,
//                    Q init, and start of softmax. PV-side MMA uses
//                    ldmatrix.x2.trans on the B operand to consume the
//                    row-major V as col-major B.
//
// Selective cp.async waits when both async knobs are on:
//   - top of tile:     wait_group<1> drains K[t]; V[t] stays in flight
//                      across QK + softmax (V load overlaps QK).
//   - end of last h:   wait_group<1> drains V[t] (or <0> on the last
//                      tile when no K[t+BC] is pending).
// vs the prior eager wait_group<0> at top, this overlap pattern lifts
// T=65K prefill 2095 → 2193 tok/s (+4.7%), with V load latency increasingly
// dominating per-tile time as T grows.
//
// Single-buffered K and V — at HD=256 a second K buffer (16 KB) doesn't fit
// the 99 KB optin shmem budget alongside Q (24 KB) and other tiles. The V
// layout flip is shmem-neutral (still HD*BC halves).
template <uint32_t HEAD_DIM, uint32_t Q_PER_KV, uint32_t BR, uint32_t BC,
          bool USE_CP_ASYNC_K, bool USE_CP_ASYNC_V, uint32_t NSPLIT>
__global__ void
__launch_bounds__(128, 1)
fattn_prefill_mma_gqa_kernel_t(
        float       * __restrict__ out,        // [batch, n_heads, HEAD_DIM] (NSPLIT==1)
        float       * __restrict__ partials,   // [n_heads, batch, NSPLIT, HEAD_DIM] (NSPLIT>1)
        float       * __restrict__ ms,         // [n_heads, batch, NSPLIT, 2]         (NSPLIT>1)
        const float * __restrict__ q,
        uint32_t     q_stride,
        const __half* __restrict__ k_cache,
        const __half* __restrict__ v_cache,
        uint32_t     n_heads,
        uint32_t     n_kv_heads,
        uint32_t     base_seq_len,
        uint32_t     batch,
        uint32_t     q_batch_stride,
        uint32_t     out_batch_stride,
        float        scale) {
    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint32_t NWARPS    = 4;
    constexpr uint32_t MMA_M     = 16;
    constexpr uint32_t MMA_N     = 8;
    constexpr uint32_t MMA_K     = 16;
    static_assert(BR == 16, "GQA MMA prefill kernel requires BR=16");
    static_assert(BC == 32, "GQA MMA prefill kernel requires BC=32");
    static_assert(HEAD_DIM == 256 || HEAD_DIM == 128,
                  "GQA MMA prefill kernel supports HEAD_DIM 128 or 256");
    static_assert(HEAD_DIM % MMA_K == 0, "HEAD_DIM must be a multiple of MMA_K");

    constexpr uint32_t QK_KSTEPS  = HEAD_DIM / MMA_K;
    constexpr uint32_t QK_N_PER_WARP = BC / NWARPS;
    static_assert(QK_N_PER_WARP == MMA_N, "BC/NWARPS must equal MMA_N");
    constexpr uint32_t PV_KSTEPS  = BC / MMA_K;
    constexpr uint32_t HD_PER_WARP = HEAD_DIM / NWARPS;
    constexpr uint32_t PV_N_TILES  = HD_PER_WARP / MMA_N;

    const uint32_t kv_head = blockIdx.x;
    const uint32_t block_q = blockIdx.y;
    const uint32_t k_split = blockIdx.z;
    const uint32_t tid     = threadIdx.x;
    const uint32_t warp    = tid / WARP_SIZE;
    const uint32_t lane    = tid % WARP_SIZE;
    if (kv_head >= n_kv_heads) return;
    const uint32_t head_base = kv_head * Q_PER_KV;

    const uint32_t group_id = lane / 4;
    const uint32_t in_group = lane % 4;

    uint32_t block_max_q = block_q * BR + BR;
    if (block_max_q > batch) block_max_q = batch;
    const uint32_t block_max_kv = base_seq_len + block_max_q;

    // Split-KV: this CTA owns [kv_lo, kv_hi) of the K axis. For NSPLIT==1 the
    // whole range is one CTA (kv_lo=0, kv_hi=block_max_kv) and the kernel
    // matches the pre-split path exactly. For NSPLIT>1 the partition is
    // BC-aligned (BC=32) so every tile in the loop touches a full BC chunk
    // and the prologue/epilogue cp.async pattern is unchanged. Out-of-range
    // tiles still produce -INFINITY scores via the existing causal mask, so
    // CTAs that overshoot block_max_kv contribute zero rows.
    uint32_t kv_lo, kv_hi;
    if (NSPLIT == 1) {
        kv_lo = 0;
        kv_hi = block_max_kv;
    } else {
        const uint32_t kv_total = base_seq_len + batch;
        const uint32_t per_split = ((kv_total + NSPLIT - 1) / NSPLIT + BC - 1) & ~uint32_t(BC - 1);
        kv_lo = k_split * per_split;
        kv_hi = kv_lo + per_split;
        if (kv_hi > block_max_kv) kv_hi = block_max_kv;
        if (kv_lo >= kv_hi) {
            // This split has no work for this q-block (entirely past the
            // causal horizon). Write a "dead" partial: m = -INF, l = 0,
            // O = 0 — combine_kernel will skip it.
            const uint32_t total_rows = Q_PER_KV * BR;
            if (tid < total_rows) {
                const uint32_t h    = tid / BR;
                const uint32_t row  = tid % BR;
                const uint32_t head = head_base + h;
                const uint32_t q_idx = block_q * BR + row;
                if (q_idx < batch) {
                    const uint64_t mb = ((uint64_t)head * batch + q_idx) * NSPLIT + k_split;
                    ms[mb * 2 + 0] = -INFINITY;
                    ms[mb * 2 + 1] = 0.0f;
                }
            }
            // Zero the partial output for every (h, row) in this block.
            // 128 threads cover Q_PER_KV*BR*HD = 6*16*256 = 24576 floats with
            // 192 floats/thread.
            for (uint32_t h = 0; h < Q_PER_KV; ++h) {
                const uint32_t head = head_base + h;
                #pragma unroll
                for (uint32_t r = 0; r < BR / NWARPS; ++r) {
                    const uint32_t q_row = warp * (BR / NWARPS) + r;
                    const uint32_t q_idx = block_q * BR + q_row;
                    if (q_idx >= batch) continue;
                    const uint64_t pbase = (((uint64_t)head * batch + q_idx)
                                            * NSPLIT + k_split) * HEAD_DIM;
                    #pragma unroll
                    for (uint32_t d = lane; d < HEAD_DIM; d += WARP_SIZE) {
                        partials[pbase + d] = 0.0f;
                    }
                }
            }
            return;
        }
    }

    // ---- Shared memory layout -------------------------------------------
    // s_V_buf is the same HD*BC halves slot used by V. When USE_CP_ASYNC_V is
    // false we view it as s_VT[d, t] (transposed, stride BC); when true we
    // view it as s_V[t, d] (row-major, stride HEAD_DIM). cp.async requires
    // the row-major layout because it can only do contiguous gmem→shmem.
    extern __shared__ char smem_raw[];
    char *p = smem_raw;
    __half *s_K     = reinterpret_cast<__half *>(p); p += BC       * HEAD_DIM * sizeof(__half);
    __half *s_V_buf = reinterpret_cast<__half *>(p); p += HEAD_DIM * BC       * sizeof(__half);
    __half *s_Q   = reinterpret_cast<__half *>(p); p += Q_PER_KV * BR * HEAD_DIM * sizeof(__half);
    float  *s_S   = reinterpret_cast<float  *>(p); p += BR       * BC       * sizeof(float);
    __half *s_P   = reinterpret_cast<__half *>(p); p += Q_PER_KV * BR * BC * sizeof(__half);
    float  *s_m   = reinterpret_cast<float  *>(p); p += Q_PER_KV * BR       * sizeof(float);
    float  *s_l   = reinterpret_cast<float  *>(p); p += Q_PER_KV * BR       * sizeof(float);
    float  *s_alpha = reinterpret_cast<float *>(p); p += BR                 * sizeof(float);

    // ---- Load Q for ALL q-heads in this kv group into s_Q ------------------
    // Layout: s_Q[h * BR * HD + r * HD + d]. 4 warps split rows.
    // Total elts = Q_PER_KV * BR * HEAD_DIM. Threads = 128.
    {
        constexpr uint32_t ROWS_PER_WARP = BR / NWARPS;          // 4
        for (uint32_t h = 0; h < Q_PER_KV; ++h) {
            const uint32_t head = head_base + h;
            for (uint32_t r = 0; r < ROWS_PER_WARP; ++r) {
                const uint32_t q_row = warp * ROWS_PER_WARP + r;
                const uint32_t q_idx = block_q * BR + q_row;
                const bool active   = (q_idx < batch);
                const float * q_ptr = active
                    ? (q + static_cast<uint64_t>(q_idx) * q_batch_stride
                         + head * q_stride)
                    : nullptr;
                #pragma unroll
                for (uint32_t d0 = 0; d0 < HEAD_DIM; d0 += WARP_SIZE) {
                    const uint32_t d = d0 + lane;
                    const float v = active ? q_ptr[d] : 0.0f;
                    s_Q[h * BR * HEAD_DIM + q_row * HEAD_DIM + d] = __float2half(v);
                }
            }
        }
    }

    // ---- Initialize per-(head,row) softmax state ---------------------------
    if (tid < Q_PER_KV * BR) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }

    // ---- Per-head output accumulator (FP32 in registers) -------------------
    float O_acc[Q_PER_KV][PV_N_TILES][4];
    #pragma unroll
    for (uint32_t h = 0; h < Q_PER_KV; ++h) {
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) O_acc[h][n][i] = 0.0f;
        }
    }

    __syncthreads();

    // K loader. issue_K(t0_target) issues cp.async (or sync loads) for the
    // K tile starting at t0_target. With cp.async, the caller is responsible
    // for the matching commit + wait_group + __syncthreads() before reading
    // s_K (the cp.async issues are non-blocking and only become visible
    // after wait_group + sync).
    auto issue_K = [&](uint32_t t0_target) {
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;
        constexpr uint32_t TOKENS_PER_PASS = 128 / LANES_PER_ROW;
        constexpr uint32_t N_PASS = BC / TOKENS_PER_PASS;
        const uint32_t row_in_pass = tid / LANES_PER_ROW;
        const uint32_t col_lane    = tid % LANES_PER_ROW;
        #pragma unroll
        for (uint32_t pass = 0; pass < N_PASS; ++pass) {
            const uint32_t shm_t = pass * TOKENS_PER_PASS + row_in_pass;
            const uint32_t t     = t0_target + shm_t;
            const uint32_t base_d = col_lane * HALVES_PER_LD;
            __half *dst = &s_K[shm_t * HEAD_DIM + base_d];
            const bool in_range = (t < block_max_kv);
            if (USE_CP_ASYNC_K) {
                if (in_range) {
                    const __half *k_src = k_cache
                        + static_cast<uint64_t>(t) * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    mma_detail::cp_async_cg_16(dst, k_src);
                } else {
                    mma_detail::cp_async_zero_16(dst);
                }
            } else {
                int4 *dst4 = reinterpret_cast<int4 *>(dst);
                if (in_range) {
                    const __half *k_src = k_cache
                        + static_cast<uint64_t>(t) * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    *dst4 = *reinterpret_cast<const int4 *>(k_src);
                } else {
                    int4 z; z.x = z.y = z.z = z.w = 0;
                    *dst4 = z;
                }
            }
        }
    };

    // V loader. issue_V(t0_target) issues cp.async (when USE_CP_ASYNC_V) or
    // performs synchronous loads (otherwise) for the V tile starting at
    // t0_target. Layouts:
    //
    //   USE_CP_ASYNC_V == false: s_V_buf is viewed as s_VT[d, t] (col-major
    //     in (d, t)) — the legacy transposed layout. The store path does a
    //     scalar-strided write per half, so cp.async cannot help.
    //
    //   USE_CP_ASYNC_V == true:  s_V_buf is viewed as s_V[t, d] (row-major
    //     in (t, d), stride HEAD_DIM). Each thread writes a contiguous
    //     16-byte chunk (8 halves), which fits cp.async.cg.16. The PV-side
    //     MMA reads it via ldmatrix.x2.trans (B operand col-major).
    //
    // Lane partitioning for both paths matches the K loader: 8 halves per
    // lane, LANES_PER_ROW lanes cover one t-row, 128 / LANES_PER_ROW rows
    // per pass.
    auto issue_V = [&](uint32_t t0_target) {
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;
        constexpr uint32_t TOKENS_PER_PASS = 128 / LANES_PER_ROW;
        constexpr uint32_t N_PASS = BC / TOKENS_PER_PASS;
        const uint32_t row_in_pass = tid / LANES_PER_ROW;
        const uint32_t col_lane    = tid % LANES_PER_ROW;
        #pragma unroll
        for (uint32_t pass = 0; pass < N_PASS; ++pass) {
            const uint32_t shm_t = pass * TOKENS_PER_PASS + row_in_pass;
            const uint32_t t     = t0_target + shm_t;
            const uint32_t base_d = col_lane * HALVES_PER_LD;
            const bool in_range = (t < block_max_kv);
            if (USE_CP_ASYNC_V) {
                __half *dst = &s_V_buf[shm_t * HEAD_DIM + base_d];
                if (in_range) {
                    const __half *v_src = v_cache
                        + static_cast<uint64_t>(t) * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    mma_detail::cp_async_cg_16(dst, v_src);
                } else {
                    mma_detail::cp_async_zero_16(dst);
                }
            } else {
                __half v8[HALVES_PER_LD];
                if (in_range) {
                    const __half *v_src = v_cache
                        + static_cast<uint64_t>(t) * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    *reinterpret_cast<int4 *>(v8) = *reinterpret_cast<const int4 *>(v_src);
                } else {
                    *reinterpret_cast<int4 *>(v8) = make_int4(0,0,0,0);
                }
                #pragma unroll
                for (uint32_t i = 0; i < HALVES_PER_LD; ++i) {
                    s_V_buf[(base_d + i) * BC + shm_t] = v8[i];
                }
            }
        }
    };

    // Prologue: issue tile 0's K and V cp.async. K commits to one group and
    // V commits to another, so we can wait on them independently if needed.
    // Order matters for the wait pattern in the loop:
    //   - K committed first (group "older")
    //   - V committed second
    //
    // wait_group<0> drains everything; wait_group<1> waits until only 1
    // group remains in flight. That lets us wait for K alone (1 outstanding
    // = the V we just committed) before QK, then wait for V before PV.
    if (USE_CP_ASYNC_K) {
        issue_K(kv_lo);
        mma_detail::cp_async_commit();
    }
    if (USE_CP_ASYNC_V) {
        issue_V(kv_lo);
        mma_detail::cp_async_commit();
    }

    // =====================================================================
    // Tile loop over KV [kv_lo, kv_hi).
    // =====================================================================
    for (uint32_t t0 = kv_lo; t0 < kv_hi; t0 += BC) {

        // ---- Selective cp.async wait at top of tile -----------------------
        // QK only reads K, so K[t] is the only async load we must drain
        // before QK. V[t] (if cp.async) can remain in flight; it's drained
        // pre-Phase-A so the V→shmem write overlaps QK.
        //
        // Commit FIFO order across the loop:
        //   prologue:  K[kv_lo], V[kv_lo]
        //   each iter: ... post-QK K[t+BC] ... post-PV V[t+BC] ...
        // So at top of tile t the two oldest are K[t], V[t] in that order,
        // and wait_group<1> drains K[t] only.
        if (USE_CP_ASYNC_K && USE_CP_ASYNC_V) {
            mma_detail::cp_async_wait_group<1>();   // drain K[t]; V[t] stays in flight
        } else if (USE_CP_ASYNC_K) {
            mma_detail::cp_async_wait_group<0>();   // K-only async: drain K[t]
        }
        // V-only async: V[t] wait is deferred to pre-Phase-A (below).

        if (!USE_CP_ASYNC_K) {
            issue_K(t0);
        }
        if (!USE_CP_ASYNC_V) {
            issue_V(t0);
        }
        __syncthreads();

        // ---- QK^T phase: compute all Q_PER_KV heads' scores in registers ----
        // Loop nest swap: K shmem reads are hoisted out of the per-head loop
        // (K doesn't depend on h), so each ks step does 1 K load and Q_PER_KV
        // Q loads + MMAs. Saves Q_PER_KV-1 K shmem reads per ks step.
        // c_h[h] holds head h's m16n8 fragment per thread (4 floats each).
        float c_h[Q_PER_KV][4];
        #pragma unroll
        for (uint32_t h = 0; h < Q_PER_KV; ++h) {
            c_h[h][0] = 0.f; c_h[h][1] = 0.f;
            c_h[h][2] = 0.f; c_h[h][3] = 0.f;
        }
        #pragma unroll
        for (uint32_t ks = 0; ks < QK_KSTEPS; ++ks) {
            const uint32_t k0 = ks * MMA_K;
            const uint32_t kv_col = warp * MMA_N + group_id;
            const __half *krow0 = &s_K[kv_col * HEAD_DIM + k0 + in_group * 2];
            const unsigned b0 = mma_detail::pack2h(krow0);
            const unsigned b1 = mma_detail::pack2h(krow0 + 8);
            #pragma unroll
            for (uint32_t h = 0; h < Q_PER_KV; ++h) {
                unsigned a0, a1, a2, a3;
                mma_detail::ldmatrix_x4_a(a0, a1, a2, a3,
                                          &s_Q[h * BR * HEAD_DIM + 0 * HEAD_DIM + k0],
                                          HEAD_DIM);
                mma_detail::mma_m16n8k16_f16f16f32(c_h[h][0], c_h[h][1], c_h[h][2], c_h[h][3],
                                                   a0, a1, a2, a3, b0, b1);
            }
        }

        // ---- Issue K[t0+BC] cp.async now (s_K is dead after QK). The
        // gmem read overlaps the upcoming Phase A softmax + Phase B PV.
        // Single-buffered K: writing s_K is safe because no one reads it
        // until the wait_group + sync at the top of the next iter. ----
        if (USE_CP_ASYNC_K) {
            const uint32_t t0_next = t0 + BC;
            if (t0_next < kv_hi) {
                issue_K(t0_next);
                mma_detail::cp_async_commit();
            }
        }

        // ---- Phase A: per-head softmax + O_acc rescale; produces s_P[h] ----
        // s_P now holds all heads simultaneously, so Phase B can share V loads.
        #pragma unroll
        for (uint32_t h = 0; h < Q_PER_KV; ++h) {

            // Write S slice to shmem with causal mask.
            {
                const uint32_t base_col = warp * MMA_N + in_group * 2;
                const uint32_t row_a = group_id;
                const uint32_t row_b = group_id + 8;
                const uint32_t q_idx_a = block_q * BR + row_a;
                const uint32_t q_idx_b = block_q * BR + row_b;
                const uint32_t my_max_kv_a = base_seq_len + q_idx_a + 1;
                const uint32_t my_max_kv_b = base_seq_len + q_idx_b + 1;
                const bool a_active = (q_idx_a < batch);
                const bool b_active = (q_idx_b < batch);
                const uint32_t t_a0 = t0 + base_col;
                const uint32_t t_a1 = t0 + base_col + 1;
                float v00 = (a_active && t_a0 < my_max_kv_a) ? c_h[h][0] * scale : -INFINITY;
                float v01 = (a_active && t_a1 < my_max_kv_a) ? c_h[h][1] * scale : -INFINITY;
                float v02 = (b_active && t_a0 < my_max_kv_b) ? c_h[h][2] * scale : -INFINITY;
                float v03 = (b_active && t_a1 < my_max_kv_b) ? c_h[h][3] * scale : -INFINITY;
                s_S[row_a * BC + base_col + 0] = v00;
                s_S[row_a * BC + base_col + 1] = v01;
                s_S[row_b * BC + base_col + 0] = v02;
                s_S[row_b * BC + base_col + 1] = v03;
            }

            __syncthreads();

            // Online softmax for this q-head's rows. Write into s_P[h*BR*BC + ...].
            {
                constexpr uint32_t ROWS_PER_WARP = BR / NWARPS;
                #pragma unroll
                for (uint32_t r = 0; r < ROWS_PER_WARP; ++r) {
                    const uint32_t row = warp * ROWS_PER_WARP + r;
                    const float s = s_S[row * BC + lane];
                    const float row_max = cuda_helpers::warp_reduce_max<32>(s);
                    const float prev_m = s_m[h * BR + row];
                    const float new_m  = fmaxf(prev_m, row_max);
                    const float p_val  = (s == -INFINITY) ? 0.0f
                                                          : __expf(s - new_m);
                    const float row_p_sum = cuda_helpers::warp_reduce_sum<32>(p_val);
                    const float prev_l   = s_l[h * BR + row];
                    const float alpha    = (prev_m == -INFINITY) ? 0.0f
                                                                 : __expf(prev_m - new_m);
                    const float new_l    = prev_l * alpha + row_p_sum;
                    if (lane == 0) {
                        s_m[h * BR + row] = new_m;
                        s_l[h * BR + row] = new_l;
                        s_alpha[row]      = alpha;
                    }
                    s_P[h * BR * BC + row * BC + lane] = __float2half(p_val);
                }
            }

            __syncthreads();

            // Rescale O_acc[h] by per-row alpha now (alpha scratch is reused next h).
            const float alpha_a = s_alpha[group_id];
            const float alpha_b = s_alpha[group_id + 8];
            #pragma unroll
            for (uint32_t n = 0; n < PV_N_TILES; ++n) {
                O_acc[h][n][0] *= alpha_a;
                O_acc[h][n][1] *= alpha_a;
                O_acc[h][n][2] *= alpha_b;
                O_acc[h][n][3] *= alpha_b;
            }

            // On the last per-head iter, drain V[t] cp.async so Phase B
            // can read s_V_buf. The shared __syncthreads below makes both
            // s_P[h_last] AND s_V_buf visible. Done here (vs at top of
            // tile) so V[t]'s gmem→shmem write overlaps QK + softmax.
            //
            // FIFO state at this point:
            //   V-only async:    {V[t]}                → wait_group<0>
            //   K+V async, mid:  {V[t], K[t+BC]}       → wait_group<1>
            //   K+V async, last: {V[t]}                → wait_group<0>
            //   V-sync:          (none — V already in shmem)
            if (h == Q_PER_KV - 1 && USE_CP_ASYNC_V) {
                const bool k_next_pending = USE_CP_ASYNC_K && (t0 + BC < kv_hi);
                if (k_next_pending) {
                    mma_detail::cp_async_wait_group<1>();
                } else {
                    mma_detail::cp_async_wait_group<0>();
                }
            }
            __syncthreads();   // s_S, s_alpha free for next head; s_P[h] visible
        }

        // ---- Phase B: PV with V loads hoisted out of per-head loop ----
        // For each (n, ks) load V once, then iterate the Q_PER_KV heads.
        // V layout depends on USE_CP_ASYNC_V: row-major s_V[t, d] with
        // ldmatrix.x2.trans, or transposed s_VT[d, t] with pack2h.
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            const uint32_t n0 = warp * HD_PER_WARP + n * MMA_N;
            #pragma unroll
            for (uint32_t ks = 0; ks < PV_KSTEPS; ++ks) {
                const uint32_t k0 = ks * MMA_K;
                unsigned b0, b1;
                if (USE_CP_ASYNC_V) {
                    // Row-major s_V[t, d]: B operand for m16n8k16.row.col
                    // is the (BC, HEAD_DIM) submatrix at (k0..k0+15, n0..n0+7),
                    // viewed col-major. ldmatrix.x2.trans does the transpose.
                    const __half *vbase = &s_V_buf[k0 * HEAD_DIM + n0];
                    mma_detail::ldmatrix_x2_b_trans(b0, b1, vbase, HEAD_DIM);
                } else {
                    // Transposed s_VT[d, t]: K stride is 1, N stride is BC.
                    const uint32_t out_col = n0 + group_id;
                    const __half *vrow0 = &s_V_buf[out_col * BC + k0 + in_group * 2];
                    b0 = mma_detail::pack2h(vrow0);
                    b1 = mma_detail::pack2h(vrow0 + 8);
                }
                #pragma unroll
                for (uint32_t h = 0; h < Q_PER_KV; ++h) {
                    unsigned a0, a1, a2, a3;
                    mma_detail::ldmatrix_x4_a(a0, a1, a2, a3,
                                              &s_P[h * BR * BC + 0 * BC + k0], BC);
                    mma_detail::mma_m16n8k16_f16f16f32(
                        O_acc[h][n][0], O_acc[h][n][1], O_acc[h][n][2], O_acc[h][n][3],
                        a0, a1, a2, a3, b0, b1);
                }
            }
        }

        // ---- Issue V[t0+BC] cp.async now (s_V_buf is dead after PV). The
        // gmem read overlaps the next iter's K wait, Q init, and softmax.
        // Single-buffered V with the wait_group + sync at the top of next
        // iter ensuring visibility (same pattern as K). ----
        if (USE_CP_ASYNC_V) {
            const uint32_t t0_next = t0 + BC;
            if (t0_next < kv_hi) {
                issue_V(t0_next);
                mma_detail::cp_async_commit();
            }
        }
    }

    // =====================================================================
    // Final write: per-head, divide by l_h and store.
    // =====================================================================
    // =====================================================================
    // Final write.
    //   NSPLIT==1: divide by l and store final output [batch, n_heads, HD].
    //   NSPLIT >1: write un-normalized O_acc partial + (m, l) per row to
    //              partials/ms scratch. The combine kernel does the cross-
    //              split online-softmax merge and writes the final output.
    // =====================================================================
    #pragma unroll
    for (uint32_t h = 0; h < Q_PER_KV; ++h) {
        const uint32_t head = head_base + h;
        const float l_a = s_l[h * BR + group_id];
        const float l_b = s_l[h * BR + group_id + 8];
        const uint32_t q_idx_a = block_q * BR + group_id;
        const uint32_t q_idx_b = block_q * BR + group_id + 8;
        if (NSPLIT == 1) {
            const float inv_la = (l_a > 0.0f) ? (1.0f / l_a) : 0.0f;
            const float inv_lb = (l_b > 0.0f) ? (1.0f / l_b) : 0.0f;
            float * const out_a = (q_idx_a < batch)
                ? (out + static_cast<uint64_t>(q_idx_a) * out_batch_stride + head * HEAD_DIM)
                : nullptr;
            float * const out_b = (q_idx_b < batch)
                ? (out + static_cast<uint64_t>(q_idx_b) * out_batch_stride + head * HEAD_DIM)
                : nullptr;
            #pragma unroll
            for (uint32_t n = 0; n < PV_N_TILES; ++n) {
                const uint32_t n0 = warp * HD_PER_WARP + n * MMA_N;
                const uint32_t col = n0 + in_group * 2;
                if (out_a) {
                    out_a[col + 0] = O_acc[h][n][0] * inv_la;
                    out_a[col + 1] = O_acc[h][n][1] * inv_la;
                }
                if (out_b) {
                    out_b[col + 0] = O_acc[h][n][2] * inv_lb;
                    out_b[col + 1] = O_acc[h][n][3] * inv_lb;
                }
            }
        } else {
            // Write un-normalized O_acc — combine kernel divides after merge.
            float * const part_a = (q_idx_a < batch)
                ? (partials + (((uint64_t)head * batch + q_idx_a) * NSPLIT + k_split) * HEAD_DIM)
                : nullptr;
            float * const part_b = (q_idx_b < batch)
                ? (partials + (((uint64_t)head * batch + q_idx_b) * NSPLIT + k_split) * HEAD_DIM)
                : nullptr;
            #pragma unroll
            for (uint32_t n = 0; n < PV_N_TILES; ++n) {
                const uint32_t n0 = warp * HD_PER_WARP + n * MMA_N;
                const uint32_t col = n0 + in_group * 2;
                if (part_a) {
                    part_a[col + 0] = O_acc[h][n][0];
                    part_a[col + 1] = O_acc[h][n][1];
                }
                if (part_b) {
                    part_b[col + 0] = O_acc[h][n][2];
                    part_b[col + 1] = O_acc[h][n][3];
                }
            }
            // Lane 0 of each group_id within warp 0 writes (m, l) for its two
            // rows. Across warp 0's 32 lanes there are 8 group_ids × 4
            // in_groups; gating on (warp==0 && in_group==0) gives exactly 8
            // writers covering rows 0..7 ("a") and 8..15 ("b") = full BR.
            if (warp == 0 && in_group == 0) {
                if (q_idx_a < batch) {
                    const uint64_t mb = ((uint64_t)head * batch + q_idx_a) * NSPLIT + k_split;
                    ms[mb * 2 + 0] = s_m[h * BR + group_id];
                    ms[mb * 2 + 1] = l_a;
                }
                if (q_idx_b < batch) {
                    const uint64_t mb = ((uint64_t)head * batch + q_idx_b) * NSPLIT + k_split;
                    ms[mb * 2 + 0] = s_m[h * BR + group_id + 8];
                    ms[mb * 2 + 1] = l_b;
                }
            }
        }
    }
}

static size_t fattn_mma_smem_bytes(uint32_t head_dim) {
    constexpr uint32_t BR = 16, BC = 32;
    size_t s = 0;
    s += BR       * head_dim * sizeof(__half);   // Q
    s += BC       * head_dim * sizeof(__half);   // K
    s += head_dim * BC       * sizeof(__half);   // V_T
    s += BR       * BC       * sizeof(float);    // S
    s += BR       * BC       * sizeof(__half);   // P
    s += BR                  * sizeof(float) * 3;// m, l, alpha
    return s;
}

static size_t fattn_mma_gqa_smem_bytes(uint32_t head_dim, uint32_t q_per_kv) {
    constexpr uint32_t BR = 16, BC = 32;
    size_t s = 0;
    s += BC       * head_dim * sizeof(__half);   // K (single, no ping-pong)
    s += head_dim * BC       * sizeof(__half);   // V_T
    s += q_per_kv * BR * head_dim * sizeof(__half);  // Q (all heads)
    s += BR       * BC       * sizeof(float);    // S
    s += q_per_kv * BR * BC  * sizeof(__half);   // P (all heads — for PV V-share)
    s += q_per_kv * BR       * sizeof(float) * 2;// m, l (per head)
    s += BR                  * sizeof(float);    // alpha (scratch)
    return s;
}

// =====================================================================
// fattn_prefill_mma_gqa_kernel_v2_t — BR=8 + ncols2=2 + Q-in-regs
// =====================================================================
//
// Mirrors llama.cpp's flash_attn_ext_f16<256,256,8,8> design but with
// ncols2=NCOLS2 chosen to match Qwen's q_per_kv exactly (2 vs llama's 8,
// avoiding 25% m-row zero-padding for q_per_kv=6).
//
// Packing: m_in_frag = c * BR + r:
//   m=0..7   → (c=0, r=0..7)
//   m=8..15  → (c=1, r=0..7)
// Lane (group_id 0..7) holds m=group_id (a-pair, c=0) and m=group_id+8
// (b-pair, c=1). One MMA m16n8k16 covers BOTH packed q-heads at once.
// Collapses v1's per-h loop entirely → 1 softmax phase per tile vs v1's 6.
//
// Shmem (HD=256, BC=64): K(32) + V(32) + S(4) + P(2) + m/l/alpha(0.2) ≈ 70 KB
// vs v1's 95 KB.
template <uint32_t HEAD_DIM, uint32_t Q_PER_KV, uint32_t BR, uint32_t BC,
          uint32_t NCOLS2, uint32_t KV_PAD,
          bool USE_CP_ASYNC_K, bool USE_CP_ASYNC_V, uint32_t NSPLIT,
          bool USE_FP16_O = false, uint32_t S_PAD = 0>
__global__ void
__launch_bounds__(128, 1)
fattn_prefill_mma_gqa_kernel_v2_t(
        float       * __restrict__ out,        // [batch, n_heads, HEAD_DIM] (NSPLIT==1)
        float       * __restrict__ partials,   // [n_heads, batch, NSPLIT, HEAD_DIM] (NSPLIT>1)
        float       * __restrict__ ms,         // [n_heads, batch, NSPLIT, 2]         (NSPLIT>1)
        const float * __restrict__ q,
        uint32_t     q_stride,
        const __half* __restrict__ k_cache,
        const __half* __restrict__ v_cache,
        uint32_t     n_heads,
        uint32_t     n_kv_heads,
        uint32_t     base_seq_len,
        uint32_t     batch,
        uint32_t     q_batch_stride,
        uint32_t     out_batch_stride,
        float        scale) {
    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint32_t NWARPS    = 4;
    constexpr uint32_t MMA_M     = 16;
    constexpr uint32_t MMA_N     = 8;
    constexpr uint32_t MMA_K     = 16;

    static_assert(BR == 8 || BR == 16 || BR == 32, "v2 supports BR ∈ {8, 16, 32}");
    static_assert(BC == 32 || BC == 64,    "v2 supports BC ∈ {32, 64}");
    static_assert(NCOLS2 == 2,             "v2 supports NCOLS2=2");
    static_assert(KV_PAD == 0 || KV_PAD == 8, "v2 supports KV_PAD ∈ {0, 8}");
    static_assert(S_PAD == 0 || S_PAD == 4, "v2 supports S_PAD ∈ {0, 4} fp32 elems");
    static_assert(BR % 8 == 0,             "BR must be multiple of 8");
    static_assert(Q_PER_KV % NCOLS2 == 0,  "Q_PER_KV must divide by NCOLS2");
    static_assert(HEAD_DIM == 128 || HEAD_DIM == 256, "v2 supports HD 128/256");
    static_assert(HEAD_DIM % MMA_K == 0,   "HEAD_DIM multiple of MMA_K");
    static_assert(BC % MMA_N == 0,         "BC multiple of MMA_N");
    static_assert(BC % NWARPS == 0,        "BC divisible by NWARPS");

    // s_S/s_P row-stride padding (in elements of their respective types).
    // S_PAD=4 fp32 elems = 16 B = 4 banks shifts each adjacent s_S row off
    // its predecessor's bank anchor, mapping the 8 group_id lanes onto
    // distinct bank quartets (was 8-way conflict on softmax-write,
    // 4-way on s_P ldmatrix.x4 in PV). +0.5 KB shmem at BR=16 BC=32;
    // still 2 blocks/SM. Same insight as KV_PAD=8 but on the score tile.
    constexpr uint32_t S_STRIDE = BC + S_PAD;          // fp32 elements
    constexpr uint32_t P_STRIDE = BC + S_PAD * 2;      // half elements (same bytes as S_PAD fp32)

    constexpr uint32_t QK_KSTEPS  = HEAD_DIM / MMA_K;          // 16
    constexpr uint32_t QK_N_TILES = (BC / NWARPS) / MMA_N;     // 2
    constexpr uint32_t PV_KSTEPS  = BC / MMA_K;                // 4
    constexpr uint32_t HD_PER_WARP = HEAD_DIM / NWARPS;        // 64
    constexpr uint32_t PV_N_TILES  = HD_PER_WARP / MMA_N;      // 8
    constexpr uint32_t Q_GROUPS_PER_KV = Q_PER_KV / NCOLS2;    // 3
    // m-axis layout. One m=16 MMA fragment packs NCOLS2=2 q-heads of
    // Q_ROWS_PER_TILE=8 q-rows each. M_TILES extends BR by stacking q-rows
    // into multiple m-tiles; each m-tile's MMA shares the same K/V B-frag
    // load → M_TILES× MMAs per shmem load (the q-rows-per-CTA lever).
    constexpr uint32_t Q_ROWS_PER_TILE = MMA_M / NCOLS2;       // 8
    constexpr uint32_t M_TILES         = BR / Q_ROWS_PER_TILE; // 1@BR=8, 2@BR=16, 4@BR=32
    constexpr uint32_t M_TOTAL         = BR * NCOLS2;          // 16@BR=8, 32@BR=16, 64@BR=32
    static_assert(M_TILES * MMA_M == M_TOTAL, "M_TILES*MMA_M==M_TOTAL");
    static_assert(M_TOTAL % NWARPS == 0,      "M_TOTAL divisible by NWARPS");

    // K/V row stride in halves. Padding by 8 halves (16 bytes) shifts each
    // adjacent row's bank-0 anchor by 4 banks, so the 8 group_id lanes that
    // share the same in-row offset land on 8 distinct banks (vs 8-way conflict
    // at KV_PAD=0). V uses HW ldmatrix.trans; the same padding still benefits
    // V because the load mask treats stride_h as the row pitch.
    constexpr uint32_t HEAD_STRIDE = HEAD_DIM + KV_PAD;

    const uint32_t kvg_idx  = blockIdx.x;
    const uint32_t kv_head  = kvg_idx / Q_GROUPS_PER_KV;
    const uint32_t q_group  = kvg_idx % Q_GROUPS_PER_KV;
    const uint32_t block_q  = blockIdx.y;
    const uint32_t k_split  = blockIdx.z;
    const uint32_t tid      = threadIdx.x;
    const uint32_t warp     = tid / WARP_SIZE;
    const uint32_t lane     = tid % WARP_SIZE;
    if (kv_head >= n_kv_heads) return;
    const uint32_t head_base = kv_head * Q_PER_KV + q_group * NCOLS2;

    const uint32_t group_id = lane / 4;
    const uint32_t in_group = lane % 4;

    constexpr uint32_t C_A = 0;
    constexpr uint32_t C_B = 1;
    const uint32_t r_a = group_id;       // m_in_frag=group_id      → (c=0, r=group_id)
    const uint32_t r_b = group_id;       // m_in_frag=group_id+8    → (c=1, r=group_id)

    uint32_t block_max_q = block_q * BR + BR;
    if (block_max_q > batch) block_max_q = batch;
    const uint32_t block_max_kv = base_seq_len + block_max_q;

    // Split-KV: this CTA owns [kv_lo, kv_hi) of the K axis (BC-aligned). For
    // NSPLIT==1 this matches the pre-split path exactly. CTAs that overshoot
    // block_max_kv produce -INFINITY scores via the existing causal mask, so
    // they contribute zero rows but still write a "live" (m, l) tagged with
    // sum=0 that combine_kernel skips. Empty splits return early via the
    // dead-partial path.
    uint32_t kv_lo, kv_hi;
    if (NSPLIT == 1) {
        kv_lo = 0;
        kv_hi = block_max_kv;
    } else {
        const uint32_t kv_total = base_seq_len + batch;
        const uint32_t per_split = ((kv_total + NSPLIT - 1) / NSPLIT + BC - 1) & ~uint32_t(BC - 1);
        kv_lo = k_split * per_split;
        kv_hi = kv_lo + per_split;
        if (kv_hi > block_max_kv) kv_hi = block_max_kv;
        if (kv_lo >= kv_hi) {
            // Dead partial: this split has no work for this q-block (entirely
            // past the causal horizon). Write m=-INF, sum=0, O=0 so combine
            // skips it.
            #pragma unroll
            for (uint32_t mt = 0; mt < M_TILES; ++mt) {
                const uint32_t q_idx_a = block_q * BR + mt * Q_ROWS_PER_TILE + r_a;
                const uint32_t q_idx_b = block_q * BR + mt * Q_ROWS_PER_TILE + r_b;
                const uint32_t head_a  = head_base + C_A;
                const uint32_t head_b  = head_base + C_B;
                if (warp == 0 && in_group == 0) {
                    if (q_idx_a < batch) {
                        const uint64_t mb = ((uint64_t)head_a * batch + q_idx_a) * NSPLIT + k_split;
                        ms[mb * 2 + 0] = -INFINITY;
                        ms[mb * 2 + 1] = 0.0f;
                    }
                    if (q_idx_b < batch) {
                        const uint64_t mb = ((uint64_t)head_b * batch + q_idx_b) * NSPLIT + k_split;
                        ms[mb * 2 + 0] = -INFINITY;
                        ms[mb * 2 + 1] = 0.0f;
                    }
                }
                #pragma unroll
                for (uint32_t n = 0; n < PV_N_TILES; ++n) {
                    const uint32_t n0 = warp * HD_PER_WARP + n * MMA_N;
                    const uint32_t col = n0 + in_group * 2;
                    if (q_idx_a < batch) {
                        const uint64_t pbase = (((uint64_t)head_a * batch + q_idx_a)
                                                * NSPLIT + k_split) * HEAD_DIM;
                        partials[pbase + col + 0] = 0.0f;
                        partials[pbase + col + 1] = 0.0f;
                    }
                    if (q_idx_b < batch) {
                        const uint64_t pbase = (((uint64_t)head_b * batch + q_idx_b)
                                                * NSPLIT + k_split) * HEAD_DIM;
                        partials[pbase + col + 0] = 0.0f;
                        partials[pbase + col + 1] = 0.0f;
                    }
                }
            }
            return;
        }
    }

    // Shmem layout. s_K/s_V_buf use HEAD_STRIDE rows = HEAD_DIM + KV_PAD halves
    // to break the bank conflict (KV_PAD=8) when enabled. Both s_K and s_V_buf
    // are BC rows × HEAD_STRIDE halves (cp.async-fill layout: [t][d]).
    extern __shared__ char smem_raw[];
    char *p = smem_raw;
    __half *s_K     = reinterpret_cast<__half *>(p); p += BC * HEAD_STRIDE * sizeof(__half);
    __half *s_V_buf = reinterpret_cast<__half *>(p); p += BC * HEAD_STRIDE * sizeof(__half);
    float  *s_S     = reinterpret_cast<float  *>(p); p += M_TOTAL * S_STRIDE * sizeof(float);
    __half *s_P     = reinterpret_cast<__half *>(p); p += M_TOTAL * P_STRIDE * sizeof(__half);
    float  *s_m     = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);
    float  *s_l     = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);
    float  *s_alpha = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);

    // ---- Q in registers --------------------------------------------------
    // Q_reg[mt][ks][slot] holds the m-tile mt's m=16 a-fragment for ks-step.
    // For BR=16, mt=0 covers q-rows 0..7, mt=1 covers q-rows 8..15.
    // Within an m-tile, slots pack two c-heads (C_A, C_B) at (r, r+8) lanes:
    //   slot 0 = pack2h(s[r_a, k0+in_group*2..+1])    head=head_base+C_A
    //   slot 1 = pack2h(s[r_b, k0+in_group*2..+1])    head=head_base+C_B
    //   slot 2 = pack2h(s[r_a, k0+in_group*2+8..+9])  head=head_base+C_A
    //   slot 3 = pack2h(s[r_b, k0+in_group*2+8..+9])  head=head_base+C_B
    unsigned Q_reg[M_TILES][QK_KSTEPS][4];
    {
        const uint32_t head_a = head_base + C_A;
        const uint32_t head_b = head_base + C_B;
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            const uint32_t q_idx_a = block_q * BR + mt * Q_ROWS_PER_TILE + r_a;
            const uint32_t q_idx_b = block_q * BR + mt * Q_ROWS_PER_TILE + r_b;
            const bool a_active = (q_idx_a < batch);
            const bool b_active = (q_idx_b < batch);
            const float *qa = a_active
                ? (q + (uint64_t)q_idx_a * q_batch_stride + head_a * q_stride) : nullptr;
            const float *qb = b_active
                ? (q + (uint64_t)q_idx_b * q_batch_stride + head_b * q_stride) : nullptr;
            #pragma unroll
            for (uint32_t ks = 0; ks < QK_KSTEPS; ++ks) {
                const uint32_t k0 = ks * MMA_K;
                const uint32_t col0 = k0 + in_group * 2;
                const __half qa0 = qa ? __float2half(qa[col0 + 0]) : (__half)0;
                const __half qa1 = qa ? __float2half(qa[col0 + 1]) : (__half)0;
                const __half qb0 = qb ? __float2half(qb[col0 + 0]) : (__half)0;
                const __half qb1 = qb ? __float2half(qb[col0 + 1]) : (__half)0;
                const __half qa8 = qa ? __float2half(qa[col0 + 8]) : (__half)0;
                const __half qa9 = qa ? __float2half(qa[col0 + 9]) : (__half)0;
                const __half qb8 = qb ? __float2half(qb[col0 + 8]) : (__half)0;
                const __half qb9 = qb ? __float2half(qb[col0 + 9]) : (__half)0;
                Q_reg[mt][ks][0] = mma_detail::pack2h(qa0, qa1);
                Q_reg[mt][ks][1] = mma_detail::pack2h(qb0, qb1);
                Q_reg[mt][ks][2] = mma_detail::pack2h(qa8, qa9);
                Q_reg[mt][ks][3] = mma_detail::pack2h(qb8, qb9);
            }
        }
    }

    if (tid < M_TOTAL) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }
    // O accumulator. USE_FP16_O=false → fp32 storage (default, parity gold).
    // USE_FP16_O=true → __half storage; MMA still f16f16f32, we promote
    // FP16 → FP32 working set, MMA, narrow back. Saves ~256 B/thread at
    // BR=16 (M_TILES*PV_N_TILES*4 fp32 → fp16). The lever is register
    // pressure for the q-rows-per-CTA bump (#5).
    using o_acc_t = typename std::conditional<USE_FP16_O, __half, float>::type;
    o_acc_t O_acc[M_TILES][PV_N_TILES][4];
    #pragma unroll
    for (uint32_t mt = 0; mt < M_TILES; ++mt) {
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) O_acc[mt][n][i] = (o_acc_t)0;
        }
    }
    __syncthreads();

    auto issue_K = [&](uint32_t t0_target) {
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;
        constexpr uint32_t TOKENS_PER_PASS = 128 / LANES_PER_ROW;
        constexpr uint32_t N_PASS = BC / TOKENS_PER_PASS;
        const uint32_t row_in_pass = tid / LANES_PER_ROW;
        const uint32_t col_lane    = tid % LANES_PER_ROW;
        #pragma unroll
        for (uint32_t pass = 0; pass < N_PASS; ++pass) {
            const uint32_t shm_t = pass * TOKENS_PER_PASS + row_in_pass;
            const uint32_t t     = t0_target + shm_t;
            const uint32_t base_d = col_lane * HALVES_PER_LD;
            __half *dst = &s_K[shm_t * HEAD_STRIDE + base_d];
            const bool in_range = (t < block_max_kv);
            if (USE_CP_ASYNC_K) {
                if (in_range) {
                    const __half *k_src = k_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    mma_detail::cp_async_cg_16(dst, k_src);
                } else {
                    mma_detail::cp_async_zero_16(dst);
                }
            } else {
                int4 *dst4 = reinterpret_cast<int4 *>(dst);
                if (in_range) {
                    const __half *k_src = k_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    *dst4 = *reinterpret_cast<const int4 *>(k_src);
                } else {
                    int4 z; z.x = z.y = z.z = z.w = 0;
                    *dst4 = z;
                }
            }
        }
    };

    auto issue_V = [&](uint32_t t0_target) {
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;
        constexpr uint32_t TOKENS_PER_PASS = 128 / LANES_PER_ROW;
        constexpr uint32_t N_PASS = BC / TOKENS_PER_PASS;
        const uint32_t row_in_pass = tid / LANES_PER_ROW;
        const uint32_t col_lane    = tid % LANES_PER_ROW;
        #pragma unroll
        for (uint32_t pass = 0; pass < N_PASS; ++pass) {
            const uint32_t shm_t = pass * TOKENS_PER_PASS + row_in_pass;
            const uint32_t t     = t0_target + shm_t;
            const uint32_t base_d = col_lane * HALVES_PER_LD;
            const bool in_range = (t < block_max_kv);
            if (USE_CP_ASYNC_V) {
                __half *dst = &s_V_buf[shm_t * HEAD_STRIDE + base_d];
                if (in_range) {
                    const __half *v_src = v_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    mma_detail::cp_async_cg_16(dst, v_src);
                } else {
                    mma_detail::cp_async_zero_16(dst);
                }
            } else {
                __half v8[HALVES_PER_LD];
                if (in_range) {
                    const __half *v_src = v_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    *reinterpret_cast<int4 *>(v8) = *reinterpret_cast<const int4 *>(v_src);
                } else {
                    *reinterpret_cast<int4 *>(v8) = make_int4(0,0,0,0);
                }
                #pragma unroll
                for (uint32_t i = 0; i < HALVES_PER_LD; ++i) {
                    s_V_buf[(base_d + i) * BC + shm_t] = v8[i];
                }
            }
        }
    };

    if (USE_CP_ASYNC_K) { issue_K(kv_lo); mma_detail::cp_async_commit(); }
    if (USE_CP_ASYNC_V) { issue_V(kv_lo); mma_detail::cp_async_commit(); }

    // ---- Tile loop -------------------------------------------------------
    for (uint32_t t0 = kv_lo; t0 < kv_hi; t0 += BC) {

        if (USE_CP_ASYNC_K && USE_CP_ASYNC_V) {
            mma_detail::cp_async_wait_group<1>();
        } else if (USE_CP_ASYNC_K) {
            mma_detail::cp_async_wait_group<0>();
        }
        if (!USE_CP_ASYNC_K) issue_K(t0);
        if (!USE_CP_ASYNC_V) issue_V(t0);
        __syncthreads();

        // QK^T phase. Each (ks, nq) loads one K B-frag and runs M_TILES MMAs
        // reusing it (the q-rows-per-CTA lever).
        float c_h[M_TILES][QK_N_TILES][4];
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            #pragma unroll
            for (uint32_t nq = 0; nq < QK_N_TILES; ++nq) {
                c_h[mt][nq][0]=0.f; c_h[mt][nq][1]=0.f;
                c_h[mt][nq][2]=0.f; c_h[mt][nq][3]=0.f;
            }
        }
        #pragma unroll
        for (uint32_t ks = 0; ks < QK_KSTEPS; ++ks) {
            const uint32_t k0 = ks * MMA_K;
            #pragma unroll
            for (uint32_t nq = 0; nq < QK_N_TILES; ++nq) {
                const uint32_t kv_col = warp * (BC / NWARPS) + nq * MMA_N + group_id;
                const __half *krow0 = &s_K[kv_col * HEAD_STRIDE + k0 + in_group * 2];
                const unsigned b0 = mma_detail::pack2h(krow0);
                const unsigned b1 = mma_detail::pack2h(krow0 + 8);
                #pragma unroll
                for (uint32_t mt = 0; mt < M_TILES; ++mt) {
                    mma_detail::mma_m16n8k16_f16f16f32(
                        c_h[mt][nq][0], c_h[mt][nq][1], c_h[mt][nq][2], c_h[mt][nq][3],
                        Q_reg[mt][ks][0], Q_reg[mt][ks][1], Q_reg[mt][ks][2], Q_reg[mt][ks][3],
                        b0, b1);
                }
            }
        }

        if (USE_CP_ASYNC_K) {
            const uint32_t t0_next = t0 + BC;
            if (t0_next < kv_hi) { issue_K(t0_next); mma_detail::cp_async_commit(); }
        }

        // Write S to shmem with causal mask. c_h[mt][nq] slots:
        //   .0/.1 → m-row m_in_frag=group_id     (c=C_A in m-tile mt, q-row mt*8+r_a)
        //   .2/.3 → m-row m_in_frag=group_id+8   (c=C_B in m-tile mt, q-row mt*8+r_b)
        // s_S layout: M_TOTAL rows × BC cols. m-tile mt occupies rows [mt*MMA_M, mt*MMA_M+16).
        {
            #pragma unroll
            for (uint32_t mt = 0; mt < M_TILES; ++mt) {
                const uint32_t q_idx_a = block_q * BR + mt * Q_ROWS_PER_TILE + r_a;
                const uint32_t q_idx_b = block_q * BR + mt * Q_ROWS_PER_TILE + r_b;
                const bool a_active = (q_idx_a < batch);
                const bool b_active = (q_idx_b < batch);
                const uint32_t my_max_kv_a = base_seq_len + q_idx_a + 1;
                const uint32_t my_max_kv_b = base_seq_len + q_idx_b + 1;
                const uint32_t m_a = mt * MMA_M + C_A * Q_ROWS_PER_TILE + r_a; // mt*16+group_id
                const uint32_t m_b = mt * MMA_M + C_B * Q_ROWS_PER_TILE + r_b; // mt*16+group_id+8
                #pragma unroll
                for (uint32_t nq = 0; nq < QK_N_TILES; ++nq) {
                    const uint32_t base_col = warp * (BC / NWARPS) + nq * MMA_N + in_group * 2;
                    const uint32_t t_a0 = t0 + base_col;
                    const uint32_t t_a1 = t0 + base_col + 1;
                    float v00 = (a_active && t_a0 < my_max_kv_a) ? c_h[mt][nq][0] * scale : -INFINITY;
                    float v01 = (a_active && t_a1 < my_max_kv_a) ? c_h[mt][nq][1] * scale : -INFINITY;
                    float v02 = (b_active && t_a0 < my_max_kv_b) ? c_h[mt][nq][2] * scale : -INFINITY;
                    float v03 = (b_active && t_a1 < my_max_kv_b) ? c_h[mt][nq][3] * scale : -INFINITY;
                    s_S[m_a * S_STRIDE + base_col + 0] = v00;
                    s_S[m_a * S_STRIDE + base_col + 1] = v01;
                    s_S[m_b * S_STRIDE + base_col + 0] = v02;
                    s_S[m_b * S_STRIDE + base_col + 1] = v03;
                }
            }
        }

        __syncthreads();

        // Online softmax over M_TOTAL rows × BC cols. NWARPS warps × ROWS_PER_WARP rows.
        // Each lane covers BC/WARP_SIZE cols.
        constexpr uint32_t ROWS_PER_WARP = M_TOTAL / NWARPS;     // 4@BR=8, 8@BR=16
        constexpr uint32_t COLS_PER_LANE = BC / WARP_SIZE;       // 1@BC=32, 2@BC=64
        #pragma unroll
        for (uint32_t r = 0; r < ROWS_PER_WARP; ++r) {
            const uint32_t m_row = warp * ROWS_PER_WARP + r;
            float local_max = -INFINITY;
            float s_vals[COLS_PER_LANE];
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                s_vals[cc] = s_S[m_row * S_STRIDE + lane * COLS_PER_LANE + cc];
                if (s_vals[cc] > local_max) local_max = s_vals[cc];
            }
            const float row_max = cuda_helpers::warp_reduce_max<32>(local_max);
            const float prev_m = s_m[m_row];
            const float new_m  = fmaxf(prev_m, row_max);
            float local_sum = 0.0f;
            __half pvals[COLS_PER_LANE];
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                const float p_val = (s_vals[cc] == -INFINITY) ? 0.0f
                                                              : __expf(s_vals[cc] - new_m);
                local_sum += p_val;
                pvals[cc] = __float2half(p_val);
            }
            const float row_sum = cuda_helpers::warp_reduce_sum<32>(local_sum);
            const float prev_l  = s_l[m_row];
            const float alpha   = (prev_m == -INFINITY) ? 0.0f : __expf(prev_m - new_m);
            const float new_l   = prev_l * alpha + row_sum;
            if (lane == 0) {
                s_m[m_row]     = new_m;
                s_l[m_row]     = new_l;
                s_alpha[m_row] = alpha;
            }
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                s_P[m_row * P_STRIDE + lane * COLS_PER_LANE + cc] = pvals[cc];
            }
        }

        if (USE_CP_ASYNC_V) {
            const bool k_next_pending = USE_CP_ASYNC_K && (t0 + BC < kv_hi);
            if (k_next_pending) mma_detail::cp_async_wait_group<1>();
            else                mma_detail::cp_async_wait_group<0>();
        }
        __syncthreads();

        // Rescale O_acc by per-row alpha. For each m-tile mt, lane group_id holds
        // m=mt*16+group_id (a-side, q-row mt*8+r_a) and m=mt*16+group_id+8 (b-side).
        float alpha_a[M_TILES], alpha_b[M_TILES];
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            alpha_a[mt] = s_alpha[mt * MMA_M + group_id];
            alpha_b[mt] = s_alpha[mt * MMA_M + group_id + 8];
        }
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            #pragma unroll
            for (uint32_t n = 0; n < PV_N_TILES; ++n) {
                const float r0 = static_cast<float>(O_acc[mt][n][0]) * alpha_a[mt];
                const float r1 = static_cast<float>(O_acc[mt][n][1]) * alpha_a[mt];
                const float r2 = static_cast<float>(O_acc[mt][n][2]) * alpha_b[mt];
                const float r3 = static_cast<float>(O_acc[mt][n][3]) * alpha_b[mt];
                O_acc[mt][n][0] = static_cast<o_acc_t>(r0);
                O_acc[mt][n][1] = static_cast<o_acc_t>(r1);
                O_acc[mt][n][2] = static_cast<o_acc_t>(r2);
                O_acc[mt][n][3] = static_cast<o_acc_t>(r3);
            }
        }

        // Phase B: PV. Each (n, ks) loads one V B-frag and runs M_TILES MMAs
        // reusing it (the q-rows-per-CTA lever). P A-frag reloaded per m-tile
        // from s_P[mt*MMA_M*P_STRIDE..].
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            const uint32_t n0 = warp * HD_PER_WARP + n * MMA_N;
            #pragma unroll
            for (uint32_t ks = 0; ks < PV_KSTEPS; ++ks) {
                const uint32_t k0 = ks * MMA_K;
                unsigned b0, b1;
                if (USE_CP_ASYNC_V) {
                    const __half *vbase = &s_V_buf[k0 * HEAD_STRIDE + n0];
                    mma_detail::ldmatrix_x2_b_trans(b0, b1, vbase, HEAD_STRIDE);
                } else {
                    const uint32_t out_col = n0 + group_id;
                    const __half *vrow0 = &s_V_buf[out_col * BC + k0 + in_group * 2];
                    b0 = mma_detail::pack2h(vrow0);
                    b1 = mma_detail::pack2h(vrow0 + 8);
                }
                #pragma unroll
                for (uint32_t mt = 0; mt < M_TILES; ++mt) {
                    unsigned a0, a1, a2, a3;
                    mma_detail::ldmatrix_x4_a(a0, a1, a2, a3,
                                              &s_P[mt * MMA_M * P_STRIDE + k0], P_STRIDE);
                    if constexpr (USE_FP16_O) {
                        float d0 = static_cast<float>(O_acc[mt][n][0]);
                        float d1 = static_cast<float>(O_acc[mt][n][1]);
                        float d2 = static_cast<float>(O_acc[mt][n][2]);
                        float d3 = static_cast<float>(O_acc[mt][n][3]);
                        mma_detail::mma_m16n8k16_f16f16f32(
                            d0, d1, d2, d3,
                            a0, a1, a2, a3, b0, b1);
                        O_acc[mt][n][0] = static_cast<o_acc_t>(d0);
                        O_acc[mt][n][1] = static_cast<o_acc_t>(d1);
                        O_acc[mt][n][2] = static_cast<o_acc_t>(d2);
                        O_acc[mt][n][3] = static_cast<o_acc_t>(d3);
                    } else {
                        float &o0 = reinterpret_cast<float &>(O_acc[mt][n][0]);
                        float &o1 = reinterpret_cast<float &>(O_acc[mt][n][1]);
                        float &o2 = reinterpret_cast<float &>(O_acc[mt][n][2]);
                        float &o3 = reinterpret_cast<float &>(O_acc[mt][n][3]);
                        mma_detail::mma_m16n8k16_f16f16f32(
                            o0, o1, o2, o3,
                            a0, a1, a2, a3, b0, b1);
                    }
                }
            }
        }

        if (USE_CP_ASYNC_V) {
            const uint32_t t0_next = t0 + BC;
            if (t0_next < kv_hi) { issue_V(t0_next); mma_detail::cp_async_commit(); }
        }
    }

    // ---- Final write -----------------------------------------------------
    // NSPLIT==1: divide by l_h and store [batch, n_heads, HEAD_DIM].
    // NSPLIT>1 : write un-normalized O_acc + (m, l) to partials/ms; the
    //            combine kernel reduces across splits and divides at the end.
    {
        const uint32_t head_a = head_base + C_A;
        const uint32_t head_b = head_base + C_B;
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            const uint32_t q_idx_a = block_q * BR + mt * Q_ROWS_PER_TILE + r_a;
            const uint32_t q_idx_b = block_q * BR + mt * Q_ROWS_PER_TILE + r_b;
            const float l_a = s_l[mt * MMA_M + C_A * Q_ROWS_PER_TILE + r_a];
            const float l_b = s_l[mt * MMA_M + C_B * Q_ROWS_PER_TILE + r_b];
            if (NSPLIT == 1) {
                const float inv_la = (l_a > 0.0f) ? (1.0f / l_a) : 0.0f;
                const float inv_lb = (l_b > 0.0f) ? (1.0f / l_b) : 0.0f;
                float * const out_a = (q_idx_a < batch)
                    ? (out + (uint64_t)q_idx_a * out_batch_stride + head_a * HEAD_DIM) : nullptr;
                float * const out_b = (q_idx_b < batch)
                    ? (out + (uint64_t)q_idx_b * out_batch_stride + head_b * HEAD_DIM) : nullptr;
                #pragma unroll
                for (uint32_t n = 0; n < PV_N_TILES; ++n) {
                    const uint32_t n0 = warp * HD_PER_WARP + n * MMA_N;
                    const uint32_t col = n0 + in_group * 2;
                    if (out_a) {
                        out_a[col + 0] = static_cast<float>(O_acc[mt][n][0]) * inv_la;
                        out_a[col + 1] = static_cast<float>(O_acc[mt][n][1]) * inv_la;
                    }
                    if (out_b) {
                        out_b[col + 0] = static_cast<float>(O_acc[mt][n][2]) * inv_lb;
                        out_b[col + 1] = static_cast<float>(O_acc[mt][n][3]) * inv_lb;
                    }
                }
            } else {
                // Un-normalized O_acc → partials. Same lane mapping as the
                // NSPLIT==1 path, just no l-divide.
                float * const part_a = (q_idx_a < batch)
                    ? (partials + (((uint64_t)head_a * batch + q_idx_a)
                                   * NSPLIT + k_split) * HEAD_DIM) : nullptr;
                float * const part_b = (q_idx_b < batch)
                    ? (partials + (((uint64_t)head_b * batch + q_idx_b)
                                   * NSPLIT + k_split) * HEAD_DIM) : nullptr;
                #pragma unroll
                for (uint32_t n = 0; n < PV_N_TILES; ++n) {
                    const uint32_t n0 = warp * HD_PER_WARP + n * MMA_N;
                    const uint32_t col = n0 + in_group * 2;
                    if (part_a) {
                        part_a[col + 0] = static_cast<float>(O_acc[mt][n][0]);
                        part_a[col + 1] = static_cast<float>(O_acc[mt][n][1]);
                    }
                    if (part_b) {
                        part_b[col + 0] = static_cast<float>(O_acc[mt][n][2]);
                        part_b[col + 1] = static_cast<float>(O_acc[mt][n][3]);
                    }
                }
                // (m, l) per (head, q_idx). Gate to a single writer per row:
                // warp 0, in_group == 0 → 8 lanes covering group_id 0..7,
                // matching r_a and r_b q-row offsets.
                if (warp == 0 && in_group == 0) {
                    const float m_a = s_m[mt * MMA_M + C_A * Q_ROWS_PER_TILE + r_a];
                    const float m_b = s_m[mt * MMA_M + C_B * Q_ROWS_PER_TILE + r_b];
                    if (q_idx_a < batch) {
                        const uint64_t mb = ((uint64_t)head_a * batch + q_idx_a) * NSPLIT + k_split;
                        ms[mb * 2 + 0] = m_a;
                        ms[mb * 2 + 1] = l_a;
                    }
                    if (q_idx_b < batch) {
                        const uint64_t mb = ((uint64_t)head_b * batch + q_idx_b) * NSPLIT + k_split;
                        ms[mb * 2 + 0] = m_b;
                        ms[mb * 2 + 1] = l_b;
                    }
                }
            }
        }
    }
}

static size_t fattn_mma_gqa_v2_smem_bytes(uint32_t head_dim, uint32_t bc, uint32_t br,
                                          uint32_t kv_pad, uint32_t s_pad) {
    constexpr uint32_t NCOLS2 = 2;
    const uint32_t M_TOTAL    = br * NCOLS2;            // 16@BR=8, 32@BR=16
    const uint32_t hd_stride  = head_dim + kv_pad;
    const uint32_t s_stride   = bc + s_pad;             // fp32 elems
    const uint32_t p_stride   = bc + s_pad * 2;         // half elems (= same bytes)
    size_t s = 0;
    s += bc       * hd_stride * sizeof(__half);   // K
    s += bc       * hd_stride * sizeof(__half);   // V
    s += M_TOTAL  * s_stride  * sizeof(float);    // S
    s += M_TOTAL  * p_stride  * sizeof(__half);   // P
    s += M_TOTAL              * sizeof(float) * 3;// m, l, alpha
    return s;
}

// =====================================================================
// fattn_prefill_mma_gqa_kernel_v3_t — BR=64 + NCOLS2=1 + Q-in-shmem
// =====================================================================
//
// NEGATIVE RESULT (2026-05-30): kept opt-in via QW3_PREFILL_ATTN=mma-gqa-v3.
// Parity-correct (top-1 disagreements within v2's own run-to-run drift), but
// throughput regressed -6%/-18%/-37% at T=4K/16K/65K vs v2 default at
// chunk=512. Two compounding losses outweigh the q-rows-per-CTA win:
//   1. 76.75 KB shmem → 1 block/SM, vs v2's 38.4 KB → 2 blocks/SM. Halved
//      occupancy means K cp.async waits stall the whole SM.
//   2. Q-from-shmem: 64 ldmatrix.x4 per tile × tiles at long T, vs v2's
//      once-per-CTA Q load into registers (zero shmem traffic for Q).
//
// Long-T attack: q-rows-per-CTA lever. v2 default is BR=16 NCOLS2=2 →
// M_TOTAL=32 q-rows/CTA, M_TILES=2 MMAs per K/V ldmatrix. llama.cpp
// flash_attn_ext_f16<256,256,8,8> packs 64 q-rows/CTA. v3 mirrors that:
//
//   BR=64, NCOLS2=1, M_TILES=4 → 64 q-rows/CTA, 4 MMAs per K/V load.
//
// Why NCOLS2=1: with BR=64 NCOLS2=2, M_TOTAL=128 doesn't fit shmem at
// HD=256 (s_S alone = 128*BC*4 ≥ 16 KB before P/Q). q_per_kv=6 isn't a
// multiple of NCOLS2=4 either. v3 keeps each m-tile single-headed; the
// grid covers all 6 q-heads via Q_GROUPS_PER_KV=6 CTAs per kv-head.
//
// Why Q-in-shmem (not regs): BR=32 NCOLS2=2 in v2 already spilled 4 KB
// at the 255-reg cap. BR=64 needs more accumulator state (O_acc=128
// fp32) so the Q_reg budget can't survive. Move Q to shmem; reload via
// ldmatrix.x4 per (mt, ks) inside the QK loop. Shmem cost: 32 KB (HD=256).
//
// Shmem (HD=256, BC=32, BR=64, NCOLS2=1):
//   s_K: 16 KB ; s_V: 16 KB ; s_Q: 32 KB ; s_S: 8 KB ; s_P: 4 KB ;
//   m/l/alpha: ~0.75 KB. Total ≈ 76.75 KB ≤ 98304-byte opt-in cap.
//   1 block/SM.
//
// Lane mapping (NCOLS2=1):
//   m_in_frag = c * BR + r:
//     m=0..7  → c=0, r=0..7 (a-pair)
//     m=8..15 → c=1, r=0..7 (b-pair)
//   For NCOLS2=1, BOTH a and b pairs target the same q-head, but
//   different q-row groups within the m-tile:
//     a-pair → q-row mt*16 + group_id           (rows 0..7 of m-tile)
//     b-pair → q-row mt*16 + group_id + 8       (rows 8..15 of m-tile)
//   One MMA m16n8k16 covers 16 q-rows in the m-axis. M_TILES=4 → 64.
//
// Q layout in shmem: row-major s_Q[BR, HD] (64 rows × 256 halves).
// ldmatrix.x4 at base=&s_Q[mt*16, ks*16] with stride=HD loads the
// canonical (m=16, k=16) a-fragment for m-tile mt step ks.
template <uint32_t HEAD_DIM, uint32_t Q_PER_KV, uint32_t BR, uint32_t BC,
          bool USE_CP_ASYNC_K, bool USE_CP_ASYNC_V>
__global__ void
__launch_bounds__(128, 1)
fattn_prefill_mma_gqa_kernel_v3_t(
        float       * __restrict__ out,
        const float * __restrict__ q,
        uint32_t     q_stride,
        const __half* __restrict__ k_cache,
        const __half* __restrict__ v_cache,
        uint32_t     n_heads,
        uint32_t     n_kv_heads,
        uint32_t     base_seq_len,
        uint32_t     batch,
        uint32_t     q_batch_stride,
        uint32_t     out_batch_stride,
        float        scale) {
    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint32_t NWARPS    = 4;
    constexpr uint32_t MMA_M     = 16;
    constexpr uint32_t MMA_N     = 8;
    constexpr uint32_t MMA_K     = 16;
    constexpr uint32_t NCOLS2    = 1;

    static_assert(BR == 32 || BR == 64,    "v3 supports BR ∈ {32, 64}");
    static_assert(BC == 32 || BC == 64,    "v3 supports BC ∈ {32, 64}");
    static_assert(BR % MMA_M == 0,         "BR must be multiple of MMA_M");
    static_assert(HEAD_DIM == 128 || HEAD_DIM == 256, "v3 supports HD 128/256");
    static_assert(HEAD_DIM % MMA_K == 0,   "HEAD_DIM multiple of MMA_K");
    static_assert(BC % MMA_N == 0,         "BC multiple of MMA_N");
    static_assert(BC % NWARPS == 0,        "BC divisible by NWARPS");

    constexpr uint32_t QK_KSTEPS  = HEAD_DIM / MMA_K;          // 16
    constexpr uint32_t QK_N_TILES = (BC / NWARPS) / MMA_N;     // 2@BC=32
    constexpr uint32_t PV_KSTEPS  = BC / MMA_K;                // 2@BC=32
    constexpr uint32_t HD_PER_WARP = HEAD_DIM / NWARPS;        // 64
    constexpr uint32_t PV_N_TILES  = HD_PER_WARP / MMA_N;      // 8
    constexpr uint32_t M_TILES     = BR / MMA_M;               // 2@BR=32, 4@BR=64
    constexpr uint32_t M_TOTAL     = BR;                       // NCOLS2=1
    static_assert(M_TOTAL % NWARPS == 0,   "M_TOTAL divisible by NWARPS");

    // Grid: blockIdx.x covers (n_kv_heads * Q_PER_KV) since each q-head is
    // its own CTA (NCOLS2=1, no head-pack into m-axis).
    const uint32_t hg_idx   = blockIdx.x;
    const uint32_t kv_head  = hg_idx / Q_PER_KV;
    const uint32_t q_in_kv  = hg_idx % Q_PER_KV;
    const uint32_t block_q  = blockIdx.y;
    const uint32_t tid      = threadIdx.x;
    const uint32_t warp     = tid / WARP_SIZE;
    const uint32_t lane     = tid % WARP_SIZE;
    if (kv_head >= n_kv_heads) return;
    const uint32_t head     = kv_head * Q_PER_KV + q_in_kv;

    const uint32_t group_id = lane / 4;
    const uint32_t in_group = lane % 4;

    uint32_t block_max_q = block_q * BR + BR;
    if (block_max_q > batch) block_max_q = batch;
    const uint32_t block_max_kv = base_seq_len + block_max_q;

    // Shmem layout.
    extern __shared__ char smem_raw[];
    char *p = smem_raw;
    __half *s_K = reinterpret_cast<__half *>(p); p += BC       * HEAD_DIM * sizeof(__half);
    __half *s_V = reinterpret_cast<__half *>(p); p += HEAD_DIM * BC       * sizeof(__half);
    __half *s_Q = reinterpret_cast<__half *>(p); p += BR       * HEAD_DIM * sizeof(__half);
    float  *s_S = reinterpret_cast<float  *>(p); p += M_TOTAL  * BC       * sizeof(float);
    __half *s_P = reinterpret_cast<__half *>(p); p += M_TOTAL  * BC       * sizeof(__half);
    float  *s_m     = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);
    float  *s_l     = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);
    float  *s_alpha = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);

    // ---- Load Q into shmem (row-major [BR, HEAD_DIM]) -------------------
    {
        // 128 threads, BR*HEAD_DIM/2 int4-equivalent halves to fill.
        // Each thread fills 8 halves (one int4) per pass.
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;          // 32@HD=256
        constexpr uint32_t ROWS_PER_PASS = 128 / LANES_PER_ROW;               // 4@HD=256
        constexpr uint32_t N_PASS        = BR / ROWS_PER_PASS;                // 16@BR=64
        const uint32_t row_in_pass = tid / LANES_PER_ROW;
        const uint32_t col_lane    = tid % LANES_PER_ROW;
        #pragma unroll
        for (uint32_t pass = 0; pass < N_PASS; ++pass) {
            const uint32_t r       = pass * ROWS_PER_PASS + row_in_pass;
            const uint32_t q_idx   = block_q * BR + r;
            const uint32_t base_d  = col_lane * HALVES_PER_LD;
            const bool active      = (q_idx < batch);
            __half *dst = &s_Q[r * HEAD_DIM + base_d];
            if (active) {
                const float *qp = q + (uint64_t)q_idx * q_batch_stride
                                + head * q_stride + base_d;
                __half h8[HALVES_PER_LD];
                #pragma unroll
                for (uint32_t i = 0; i < HALVES_PER_LD; ++i) {
                    h8[i] = __float2half(qp[i]);
                }
                *reinterpret_cast<int4 *>(dst) = *reinterpret_cast<const int4 *>(h8);
            } else {
                int4 z; z.x = z.y = z.z = z.w = 0;
                *reinterpret_cast<int4 *>(dst) = z;
            }
        }
    }

    if (tid < M_TOTAL) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }
    float O_acc[M_TILES][PV_N_TILES][4];
    #pragma unroll
    for (uint32_t mt = 0; mt < M_TILES; ++mt) {
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) O_acc[mt][n][i] = 0.0f;
        }
    }
    __syncthreads();

    auto issue_K = [&](uint32_t t0_target) {
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;
        constexpr uint32_t TOKENS_PER_PASS = 128 / LANES_PER_ROW;
        constexpr uint32_t N_PASS = BC / TOKENS_PER_PASS;
        const uint32_t row_in_pass = tid / LANES_PER_ROW;
        const uint32_t col_lane    = tid % LANES_PER_ROW;
        #pragma unroll
        for (uint32_t pass = 0; pass < N_PASS; ++pass) {
            const uint32_t shm_t = pass * TOKENS_PER_PASS + row_in_pass;
            const uint32_t t     = t0_target + shm_t;
            const uint32_t base_d = col_lane * HALVES_PER_LD;
            __half *dst = &s_K[shm_t * HEAD_DIM + base_d];
            const bool in_range = (t < block_max_kv);
            if (USE_CP_ASYNC_K) {
                if (in_range) {
                    const __half *k_src = k_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    mma_detail::cp_async_cg_16(dst, k_src);
                } else {
                    mma_detail::cp_async_zero_16(dst);
                }
            } else {
                int4 *dst4 = reinterpret_cast<int4 *>(dst);
                if (in_range) {
                    const __half *k_src = k_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    *dst4 = *reinterpret_cast<const int4 *>(k_src);
                } else {
                    int4 z; z.x = z.y = z.z = z.w = 0;
                    *dst4 = z;
                }
            }
        }
    };

    auto issue_V = [&](uint32_t t0_target) {
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;
        constexpr uint32_t TOKENS_PER_PASS = 128 / LANES_PER_ROW;
        constexpr uint32_t N_PASS = BC / TOKENS_PER_PASS;
        const uint32_t row_in_pass = tid / LANES_PER_ROW;
        const uint32_t col_lane    = tid % LANES_PER_ROW;
        #pragma unroll
        for (uint32_t pass = 0; pass < N_PASS; ++pass) {
            const uint32_t shm_t = pass * TOKENS_PER_PASS + row_in_pass;
            const uint32_t t     = t0_target + shm_t;
            const uint32_t base_d = col_lane * HALVES_PER_LD;
            const bool in_range = (t < block_max_kv);
            if (USE_CP_ASYNC_V) {
                __half *dst = &s_V[shm_t * HEAD_DIM + base_d];
                if (in_range) {
                    const __half *v_src = v_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    mma_detail::cp_async_cg_16(dst, v_src);
                } else {
                    mma_detail::cp_async_zero_16(dst);
                }
            } else {
                __half v8[HALVES_PER_LD];
                if (in_range) {
                    const __half *v_src = v_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    *reinterpret_cast<int4 *>(v8) = *reinterpret_cast<const int4 *>(v_src);
                } else {
                    *reinterpret_cast<int4 *>(v8) = make_int4(0,0,0,0);
                }
                #pragma unroll
                for (uint32_t i = 0; i < HALVES_PER_LD; ++i) {
                    s_V[(base_d + i) * BC + shm_t] = v8[i];
                }
            }
        }
    };

    if (USE_CP_ASYNC_K) { issue_K(0); mma_detail::cp_async_commit(); }
    if (USE_CP_ASYNC_V) { issue_V(0); mma_detail::cp_async_commit(); }
    const uint32_t kv_hi = block_max_kv;

    // ---- Tile loop -------------------------------------------------------
    for (uint32_t t0 = 0; t0 < kv_hi; t0 += BC) {

        if (USE_CP_ASYNC_K && USE_CP_ASYNC_V) {
            mma_detail::cp_async_wait_group<1>();
        } else if (USE_CP_ASYNC_K) {
            mma_detail::cp_async_wait_group<0>();
        }
        if (!USE_CP_ASYNC_K) issue_K(t0);
        if (!USE_CP_ASYNC_V) issue_V(t0);
        __syncthreads();

        // QK^T phase. Each (ks, nq) loads one K B-frag and runs M_TILES MMAs
        // reusing it (the q-rows-per-CTA lever).
        // For NCOLS2=1, both a-pair and b-pair within an m-tile target the
        // same q-head, different q-row groups. Q is loaded via ldmatrix.x4
        // from s_Q[mt*16 .. mt*16+15, ks*16 .. ks*16+15].
        float c_h[M_TILES][QK_N_TILES][4];
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            #pragma unroll
            for (uint32_t nq = 0; nq < QK_N_TILES; ++nq) {
                c_h[mt][nq][0]=0.f; c_h[mt][nq][1]=0.f;
                c_h[mt][nq][2]=0.f; c_h[mt][nq][3]=0.f;
            }
        }
        #pragma unroll
        for (uint32_t ks = 0; ks < QK_KSTEPS; ++ks) {
            const uint32_t k0 = ks * MMA_K;
            // Load K B-frags for all QK_N_TILES.
            unsigned bb[QK_N_TILES][2];
            #pragma unroll
            for (uint32_t nq = 0; nq < QK_N_TILES; ++nq) {
                const uint32_t kv_col = warp * (BC / NWARPS) + nq * MMA_N + group_id;
                const __half *krow0 = &s_K[kv_col * HEAD_DIM + k0 + in_group * 2];
                bb[nq][0] = mma_detail::pack2h(krow0);
                bb[nq][1] = mma_detail::pack2h(krow0 + 8);
            }
            // Per m-tile: ldmatrix.x4 the Q a-frag, then run MMAs for all nq.
            #pragma unroll
            for (uint32_t mt = 0; mt < M_TILES; ++mt) {
                unsigned a0, a1, a2, a3;
                mma_detail::ldmatrix_x4_a(a0, a1, a2, a3,
                                          &s_Q[mt * MMA_M * HEAD_DIM + k0],
                                          HEAD_DIM);
                #pragma unroll
                for (uint32_t nq = 0; nq < QK_N_TILES; ++nq) {
                    mma_detail::mma_m16n8k16_f16f16f32(
                        c_h[mt][nq][0], c_h[mt][nq][1], c_h[mt][nq][2], c_h[mt][nq][3],
                        a0, a1, a2, a3,
                        bb[nq][0], bb[nq][1]);
                }
            }
        }

        if (USE_CP_ASYNC_K) {
            const uint32_t t0_next = t0 + BC;
            if (t0_next < kv_hi) { issue_K(t0_next); mma_detail::cp_async_commit(); }
        }

        // Write S to shmem with causal mask. c_h[mt][nq] slots:
        //   .0/.1 → m-row m_in_frag=group_id     (q-row mt*16+group_id)
        //   .2/.3 → m-row m_in_frag=group_id+8   (q-row mt*16+group_id+8)
        // s_S layout: M_TOTAL rows × BC cols. m-tile mt occupies rows
        // [mt*MMA_M, mt*MMA_M+16).
        {
            #pragma unroll
            for (uint32_t mt = 0; mt < M_TILES; ++mt) {
                const uint32_t q_idx_a = block_q * BR + mt * MMA_M + group_id;
                const uint32_t q_idx_b = block_q * BR + mt * MMA_M + group_id + 8;
                const bool a_active = (q_idx_a < batch);
                const bool b_active = (q_idx_b < batch);
                const uint32_t my_max_kv_a = base_seq_len + q_idx_a + 1;
                const uint32_t my_max_kv_b = base_seq_len + q_idx_b + 1;
                const uint32_t m_a = mt * MMA_M + group_id;
                const uint32_t m_b = mt * MMA_M + group_id + 8;
                #pragma unroll
                for (uint32_t nq = 0; nq < QK_N_TILES; ++nq) {
                    const uint32_t base_col = warp * (BC / NWARPS) + nq * MMA_N + in_group * 2;
                    const uint32_t t_a0 = t0 + base_col;
                    const uint32_t t_a1 = t0 + base_col + 1;
                    float v00 = (a_active && t_a0 < my_max_kv_a) ? c_h[mt][nq][0] * scale : -INFINITY;
                    float v01 = (a_active && t_a1 < my_max_kv_a) ? c_h[mt][nq][1] * scale : -INFINITY;
                    float v02 = (b_active && t_a0 < my_max_kv_b) ? c_h[mt][nq][2] * scale : -INFINITY;
                    float v03 = (b_active && t_a1 < my_max_kv_b) ? c_h[mt][nq][3] * scale : -INFINITY;
                    s_S[m_a * BC + base_col + 0] = v00;
                    s_S[m_a * BC + base_col + 1] = v01;
                    s_S[m_b * BC + base_col + 0] = v02;
                    s_S[m_b * BC + base_col + 1] = v03;
                }
            }
        }

        __syncthreads();

        // Online softmax over M_TOTAL rows × BC cols.
        constexpr uint32_t ROWS_PER_WARP = M_TOTAL / NWARPS;     // 16@BR=64
        constexpr uint32_t COLS_PER_LANE = BC / WARP_SIZE;       // 1@BC=32, 2@BC=64
        #pragma unroll
        for (uint32_t r = 0; r < ROWS_PER_WARP; ++r) {
            const uint32_t m_row = warp * ROWS_PER_WARP + r;
            float local_max = -INFINITY;
            float s_vals[COLS_PER_LANE];
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                s_vals[cc] = s_S[m_row * BC + lane * COLS_PER_LANE + cc];
                if (s_vals[cc] > local_max) local_max = s_vals[cc];
            }
            const float row_max = cuda_helpers::warp_reduce_max<32>(local_max);
            const float prev_m = s_m[m_row];
            const float new_m  = fmaxf(prev_m, row_max);
            float local_sum = 0.0f;
            __half pvals[COLS_PER_LANE];
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                const float p_val = (s_vals[cc] == -INFINITY) ? 0.0f
                                                              : __expf(s_vals[cc] - new_m);
                local_sum += p_val;
                pvals[cc] = __float2half(p_val);
            }
            const float row_sum = cuda_helpers::warp_reduce_sum<32>(local_sum);
            const float prev_l  = s_l[m_row];
            const float alpha   = (prev_m == -INFINITY) ? 0.0f : __expf(prev_m - new_m);
            const float new_l   = prev_l * alpha + row_sum;
            if (lane == 0) {
                s_m[m_row]     = new_m;
                s_l[m_row]     = new_l;
                s_alpha[m_row] = alpha;
            }
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                s_P[m_row * BC + lane * COLS_PER_LANE + cc] = pvals[cc];
            }
        }

        if (USE_CP_ASYNC_V) {
            const bool k_next_pending = USE_CP_ASYNC_K && (t0 + BC < kv_hi);
            if (k_next_pending) mma_detail::cp_async_wait_group<1>();
            else                mma_detail::cp_async_wait_group<0>();
        }
        __syncthreads();

        // Rescale O_acc by per-row alpha. For each m-tile mt, lane group_id
        // holds m=mt*16+group_id (a-side, q-row mt*16+group_id) and
        // m=mt*16+group_id+8 (b-side, q-row mt*16+group_id+8).
        float alpha_a[M_TILES], alpha_b[M_TILES];
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            alpha_a[mt] = s_alpha[mt * MMA_M + group_id];
            alpha_b[mt] = s_alpha[mt * MMA_M + group_id + 8];
        }
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            #pragma unroll
            for (uint32_t n = 0; n < PV_N_TILES; ++n) {
                O_acc[mt][n][0] *= alpha_a[mt];
                O_acc[mt][n][1] *= alpha_a[mt];
                O_acc[mt][n][2] *= alpha_b[mt];
                O_acc[mt][n][3] *= alpha_b[mt];
            }
        }

        // Phase B: PV. Each (n, ks) loads one V B-frag and runs M_TILES MMAs
        // reusing it (the q-rows-per-CTA lever).
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            const uint32_t n0 = warp * HD_PER_WARP + n * MMA_N;
            #pragma unroll
            for (uint32_t ks = 0; ks < PV_KSTEPS; ++ks) {
                const uint32_t k0 = ks * MMA_K;
                unsigned b0, b1;
                if (USE_CP_ASYNC_V) {
                    const __half *vbase = &s_V[k0 * HEAD_DIM + n0];
                    mma_detail::ldmatrix_x2_b_trans(b0, b1, vbase, HEAD_DIM);
                } else {
                    const uint32_t out_col = n0 + group_id;
                    const __half *vrow0 = &s_V[out_col * BC + k0 + in_group * 2];
                    b0 = mma_detail::pack2h(vrow0);
                    b1 = mma_detail::pack2h(vrow0 + 8);
                }
                #pragma unroll
                for (uint32_t mt = 0; mt < M_TILES; ++mt) {
                    unsigned a0, a1, a2, a3;
                    mma_detail::ldmatrix_x4_a(a0, a1, a2, a3,
                                              &s_P[mt * MMA_M * BC + k0], BC);
                    mma_detail::mma_m16n8k16_f16f16f32(
                        O_acc[mt][n][0], O_acc[mt][n][1], O_acc[mt][n][2], O_acc[mt][n][3],
                        a0, a1, a2, a3, b0, b1);
                }
            }
        }

        if (USE_CP_ASYNC_V) {
            const uint32_t t0_next = t0 + BC;
            if (t0_next < kv_hi) { issue_V(t0_next); mma_detail::cp_async_commit(); }
        }
    }

    // ---- Final write: divide by l and store [batch, n_heads, HEAD_DIM] --
    {
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            const uint32_t q_idx_a = block_q * BR + mt * MMA_M + group_id;
            const uint32_t q_idx_b = block_q * BR + mt * MMA_M + group_id + 8;
            const float l_a = s_l[mt * MMA_M + group_id];
            const float l_b = s_l[mt * MMA_M + group_id + 8];
            const float inv_la = (l_a > 0.0f) ? (1.0f / l_a) : 0.0f;
            const float inv_lb = (l_b > 0.0f) ? (1.0f / l_b) : 0.0f;
            float * const out_a = (q_idx_a < batch)
                ? (out + (uint64_t)q_idx_a * out_batch_stride + head * HEAD_DIM) : nullptr;
            float * const out_b = (q_idx_b < batch)
                ? (out + (uint64_t)q_idx_b * out_batch_stride + head * HEAD_DIM) : nullptr;
            #pragma unroll
            for (uint32_t n = 0; n < PV_N_TILES; ++n) {
                const uint32_t n0 = warp * HD_PER_WARP + n * MMA_N;
                const uint32_t col = n0 + in_group * 2;
                if (out_a) {
                    out_a[col + 0] = O_acc[mt][n][0] * inv_la;
                    out_a[col + 1] = O_acc[mt][n][1] * inv_la;
                }
                if (out_b) {
                    out_b[col + 0] = O_acc[mt][n][2] * inv_lb;
                    out_b[col + 1] = O_acc[mt][n][3] * inv_lb;
                }
            }
        }
    }
}

static size_t fattn_mma_gqa_v3_smem_bytes(uint32_t head_dim, uint32_t bc, uint32_t br) {
    const uint32_t M_TOTAL = br;                     // NCOLS2=1
    size_t s = 0;
    s += bc       * head_dim * sizeof(__half);   // K
    s += head_dim * bc       * sizeof(__half);   // V
    s += br       * head_dim * sizeof(__half);   // Q (in shmem)
    s += M_TOTAL  * bc       * sizeof(float);    // S
    s += M_TOTAL  * bc       * sizeof(__half);   // P
    s += M_TOTAL             * sizeof(float) * 3;// m, l, alpha
    return s;
}


// =====================================================================
// fattn_prefill_mma_gqa_kernel_v4_t — NWARPS=8 + warp-pair-owns-mtile
// =====================================================================
//
// Long-T attack #2 (after v3 negative). Mirrors llama.cpp's M_TOTAL=64
// q-rows-per-CTA exactly while keeping per-thread Q in registers.
//
// Geometry:
//   NWARPS = 8 (vs v2's 4), 256 threads/CTA
//   BR = 8, BC = 32, NCOLS2 = 2, M_TILES = 4 → M_TOTAL = 64
//   warp_pair = warp / 2 ∈ {0..3} → owns its own m-tile (8 q-tokens × 2 q-heads)
//   warp_in_pair = warp % 2 ∈ {0,1} → owns its BC half (16 cols × 2 MMA_N tiles)
//
// q-head pad: gqa=6 padded to 8 in m-axis. Heads 6,7 produce -INF scores
// (masked out before softmax) → zero contribution to O. M_TOTAL=64 active
// rows of which 8/64 = 12.5% are zero-padded compute (still cheaper than
// re-reading K/V to avoid the pad).
//
// Why warp-pair-owns-mtile (vs v2's all-warps-share-mtiles):
//   v2 NWARPS=4 with M_TILES=4 (BR=32 NCOLS2=2) hits the 255-reg cap
//   because every warp holds Q_reg for ALL 4 m-tiles (256 fp32/thread for
//   Q alone). v3 (BR=64 NCOLS2=1, NWARPS=4) moved Q to shmem to escape;
//   the 32 KB s_Q + ldmatrix.x4 per (mt, ks) cost regressed throughput.
//   v4 keeps Q in registers but partitions m-tiles ACROSS warp pairs:
//   each warp holds Q only for its own m-tile (64 fp32/thread, well
//   under cap).
//
// Why NWARPS=8 (vs v2's 4):
//   With 4 m-tiles × 2 warps each = 8 warps. Each warp pair splits the
//   BC axis (16 cols/warp = 2 MMA_N tiles). NWARPS=4 with warp-pair
//   structure would need M_TILES=2 (just BR=16 NCOLS2=2 = v2 default,
//   no win), so the 8-warp count is forced by the M_TOTAL=64 target.
//
// Per-thread register footprint (vs v2 default BR=16 NCOLS2=2):
//   Q_reg:  v2  M_TILES=2 × KSTEPS=16 × 4   = 128 fp32
//           v4  M_TILES_PER_WARP=1 × 16 × 4 = 64 fp32  (-50%)
//   c_h:    v2  2 × QK_N_TILES=2 × 4        = 16 fp32  (per warp)
//           v4  1 × 2 × 4                    = 8 fp32   (per warp)
//   O_acc:  v2  2 × PV_N_TILES=8 × 4        = 64 fp32
//           v4  1 × PV_N_TILES=16 × 4       = 64 fp32  (HD/2 cols/warp)
//   Net Q+c_h+O ≈ 136 fp32/thread (v2 def: 208). Predicted ~150-180 reg32
//   per thread, 0 spills.
//
// Shmem (HD=256, BC=32, BR=8, NCOLS2=2, M_TILES=4, KV_PAD=8, S_PAD=4):
//   s_K: 32×264×2 = 16.5 KB ; s_V: 16.5 KB
//   s_S: 64×36×4 = 9 KB     ; s_P: 64×40×2 = 5 KB
//   s_m, s_l, s_alpha: 64×4×3 = 0.75 KB
//   Total ≈ 47.7 KB — 2 blocks/SM at sm_120a 100 KB opt-in cap.
//
// Per-tile sync count (BC=32):
//   1× post-K-load (post-cp.async wait)
//   1× post-QK-write s_S (before softmax read)
//   1× post-softmax write s_P (before PV ldmatrix)
//   1× pre-V-load wait_group<0>
//   = 4 syncs/tile (v2 default = 3 syncs/tile; v4's 1 extra is the cost
//   of warp-pair partial-S merge implicit in s_S).
//
// Grid:
//   (n_kv_heads, n_blocks_q, NSPLIT)
//   block.x = NWARPS * 32 = 256
template <uint32_t HEAD_DIM,
          bool USE_CP_ASYNC_K, bool USE_CP_ASYNC_V, uint32_t NSPLIT,
          bool USE_FP16_O = false>
__global__ void
__launch_bounds__(256, 1)
fattn_prefill_mma_gqa_kernel_v4_t(
        float       * __restrict__ out,
        float       * __restrict__ partials,    // NSPLIT>1 only
        float       * __restrict__ ms,          // NSPLIT>1 only: [n_heads*batch*NSPLIT*2]
        const float * __restrict__ q,
        uint32_t     q_stride,
        const __half* __restrict__ k_cache,
        const __half* __restrict__ v_cache,
        uint32_t     n_heads,
        uint32_t     n_kv_heads,
        uint32_t     base_seq_len,
        uint32_t     batch,
        uint32_t     q_batch_stride,
        uint32_t     out_batch_stride,
        float        scale) {
    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint32_t NWARPS    = 8;
    constexpr uint32_t MMA_M     = 16;
    constexpr uint32_t MMA_N     = 8;
    constexpr uint32_t MMA_K     = 16;
    constexpr uint32_t Q_PER_KV  = 6;     // active heads (gqa=6)
    constexpr uint32_t Q_PADDED  = 8;     // padded to 8 = NCOLS2 * M_TILES
    constexpr uint32_t BR        = 8;
    constexpr uint32_t BC        = 32;
    constexpr uint32_t NCOLS2    = 2;
    constexpr uint32_t M_TILES   = 4;     // 4 warp pairs × 1 m-tile/pair
    constexpr uint32_t M_TOTAL   = M_TILES * MMA_M;       // 64
    constexpr uint32_t KV_PAD    = 8;
    constexpr uint32_t S_PAD     = 4;     // fp32 elems
    constexpr uint32_t S_STRIDE  = BC + S_PAD;            // 36 fp32
    constexpr uint32_t P_STRIDE  = BC + S_PAD * 2;        // 40 half (=160 B = same byte width as S_STRIDE fp32)
    constexpr uint32_t HEAD_STRIDE = HEAD_DIM + KV_PAD;   // 264

    static_assert(HEAD_DIM == 128 || HEAD_DIM == 256, "v4 supports HD 128/256");
    static_assert(M_TOTAL == NCOLS2 * M_TILES * MMA_M / NCOLS2, "M_TOTAL consistency");
    static_assert(NWARPS == 2 * M_TILES, "v4: 2 warps per m-tile");

    constexpr uint32_t WARPS_PER_PAIR     = 2;
    constexpr uint32_t COLS_PER_WARP      = BC / WARPS_PER_PAIR;       // 16
    constexpr uint32_t QK_N_TILES_PER_WARP = COLS_PER_WARP / MMA_N;    // 2
    constexpr uint32_t QK_KSTEPS          = HEAD_DIM / MMA_K;          // 16
    constexpr uint32_t HD_PER_WARP_IN_PAIR = HEAD_DIM / WARPS_PER_PAIR;// 128
    constexpr uint32_t PV_N_TILES         = HD_PER_WARP_IN_PAIR / MMA_N; // 16
    constexpr uint32_t PV_KSTEPS          = BC / MMA_K;                // 2
    constexpr uint32_t Q_ROWS_PER_TILE    = MMA_M / NCOLS2;            // 8 = BR

    // Grid: each kv-head is one CTA along blockIdx.x. The CTA covers all
    // Q_PADDED=8 q-heads of that kv-head (6 active + 2 zero-padded) packed
    // into M_TILES=4 m-tiles. So Q_GROUPS_PER_KV = 1 here (vs v2's 3).
    const uint32_t kv_head = blockIdx.x;
    const uint32_t block_q = blockIdx.y;
    const uint32_t k_split = blockIdx.z;
    const uint32_t tid     = threadIdx.x;
    const uint32_t warp    = tid / WARP_SIZE;
    const uint32_t lane    = tid % WARP_SIZE;
    if (kv_head >= n_kv_heads) return;

    const uint32_t warp_pair    = warp / WARPS_PER_PAIR;   // 0..3 = mt
    const uint32_t warp_in_pair = warp % WARPS_PER_PAIR;   // 0,1
    const uint32_t mt           = warp_pair;
    const uint32_t my_bc_lo     = warp_in_pair * COLS_PER_WARP;

    const uint32_t group_id = lane / 4;
    const uint32_t in_group = lane % 4;

    constexpr uint32_t C_A = 0;
    constexpr uint32_t C_B = 1;
    const uint32_t r_a = group_id;       // q-row within tile, a-pair
    const uint32_t r_b = group_id;       // q-row within tile, b-pair

    // Active heads for this CTA: head_base + 0..5 active, head_base+6..7 padded.
    const uint32_t head_base = kv_head * Q_PER_KV;
    // For m-tile mt, this warp-pair owns 2 q-heads (NCOLS2=2):
    //   head_a_idx = mt * 2 + 0 (in [0..7] padded space)
    //   head_b_idx = mt * 2 + 1
    const uint32_t head_a_idx_in_kv = mt * NCOLS2 + C_A;
    const uint32_t head_b_idx_in_kv = mt * NCOLS2 + C_B;
    const bool head_a_active = (head_a_idx_in_kv < Q_PER_KV);
    const bool head_b_active = (head_b_idx_in_kv < Q_PER_KV);
    const uint32_t head_a = head_base + head_a_idx_in_kv;
    const uint32_t head_b = head_base + head_b_idx_in_kv;

    uint32_t block_max_q = block_q * BR + BR;
    if (block_max_q > batch) block_max_q = batch;
    const uint32_t block_max_kv = base_seq_len + block_max_q;

    // Split-KV.
    uint32_t kv_lo, kv_hi;
    if (NSPLIT == 1) {
        kv_lo = 0;
        kv_hi = block_max_kv;
    } else {
        const uint32_t kv_total = base_seq_len + batch;
        const uint32_t per_split = ((kv_total + NSPLIT - 1) / NSPLIT + BC - 1) & ~uint32_t(BC - 1);
        kv_lo = k_split * per_split;
        kv_hi = kv_lo + per_split;
        if (kv_hi > block_max_kv) kv_hi = block_max_kv;
        if (kv_lo >= kv_hi) {
            // Dead partial.
            const uint32_t q_idx_a = block_q * BR + r_a;
            const uint32_t q_idx_b = block_q * BR + r_b;
            if (warp_in_pair == 0 && in_group == 0) {
                if (head_a_active && q_idx_a < batch) {
                    const uint64_t mb = ((uint64_t)head_a * batch + q_idx_a) * NSPLIT + k_split;
                    ms[mb * 2 + 0] = -INFINITY;
                    ms[mb * 2 + 1] = 0.0f;
                }
                if (head_b_active && q_idx_b < batch) {
                    const uint64_t mb = ((uint64_t)head_b * batch + q_idx_b) * NSPLIT + k_split;
                    ms[mb * 2 + 0] = -INFINITY;
                    ms[mb * 2 + 1] = 0.0f;
                }
            }
            #pragma unroll
            for (uint32_t n = 0; n < PV_N_TILES; ++n) {
                const uint32_t n0 = warp_in_pair * HD_PER_WARP_IN_PAIR + n * MMA_N;
                const uint32_t col = n0 + in_group * 2;
                const uint64_t pa = ((uint64_t)head_a * batch + q_idx_a) * NSPLIT + k_split;
                const uint64_t pb = ((uint64_t)head_b * batch + q_idx_b) * NSPLIT + k_split;
                if (head_a_active && q_idx_a < batch) {
                    partials[pa * HEAD_DIM + col + 0] = 0.0f;
                    partials[pa * HEAD_DIM + col + 1] = 0.0f;
                }
                if (head_b_active && q_idx_b < batch) {
                    partials[pb * HEAD_DIM + col + 0] = 0.0f;
                    partials[pb * HEAD_DIM + col + 1] = 0.0f;
                }
            }
            return;
        }
    }

    // Shmem.
    extern __shared__ char smem_raw[];
    char *p = smem_raw;
    __half *s_K     = reinterpret_cast<__half *>(p); p += BC * HEAD_STRIDE * sizeof(__half);
    __half *s_V_buf = reinterpret_cast<__half *>(p); p += BC * HEAD_STRIDE * sizeof(__half);
    float  *s_S     = reinterpret_cast<float  *>(p); p += M_TOTAL * S_STRIDE * sizeof(float);
    __half *s_P     = reinterpret_cast<__half *>(p); p += M_TOTAL * P_STRIDE * sizeof(__half);
    float  *s_m     = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);
    float  *s_l     = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);
    float  *s_alpha = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);
    (void)s_alpha;

    // ---- Q registers (one m-tile per warp pair, replicated across the pair).
    // Layout per thread: Q_reg[QK_KSTEPS][4]; the 4 slots are
    // (a0_lo, b0_lo, a0_hi, b0_hi) packed-half pairs that match the
    // ldmatrix.x4 a-fragment layout used in mma_m16n8k16_f16f16f32.
    unsigned Q_reg[QK_KSTEPS][4];
    {
        const uint32_t q_idx_a = block_q * BR + r_a;
        const uint32_t q_idx_b = block_q * BR + r_b;
        const bool a_active = head_a_active && (q_idx_a < batch);
        const bool b_active = head_b_active && (q_idx_b < batch);
        const float *qa = a_active
            ? (q + (uint64_t)q_idx_a * q_batch_stride + head_a * q_stride) : nullptr;
        const float *qb = b_active
            ? (q + (uint64_t)q_idx_b * q_batch_stride + head_b * q_stride) : nullptr;
        #pragma unroll
        for (uint32_t ks = 0; ks < QK_KSTEPS; ++ks) {
            const uint32_t k0 = ks * MMA_K;
            const uint32_t col0 = k0 + in_group * 2;
            const __half qa0 = qa ? __float2half(qa[col0 + 0]) : (__half)0;
            const __half qa1 = qa ? __float2half(qa[col0 + 1]) : (__half)0;
            const __half qb0 = qb ? __float2half(qb[col0 + 0]) : (__half)0;
            const __half qb1 = qb ? __float2half(qb[col0 + 1]) : (__half)0;
            const __half qa8 = qa ? __float2half(qa[col0 + 8]) : (__half)0;
            const __half qa9 = qa ? __float2half(qa[col0 + 9]) : (__half)0;
            const __half qb8 = qb ? __float2half(qb[col0 + 8]) : (__half)0;
            const __half qb9 = qb ? __float2half(qb[col0 + 9]) : (__half)0;
            Q_reg[ks][0] = mma_detail::pack2h(qa0, qa1);
            Q_reg[ks][1] = mma_detail::pack2h(qb0, qb1);
            Q_reg[ks][2] = mma_detail::pack2h(qa8, qa9);
            Q_reg[ks][3] = mma_detail::pack2h(qb8, qb9);
        }
    }

    if (tid < M_TOTAL) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }

    using o_acc_t = typename std::conditional<USE_FP16_O, __half, float>::type;
    o_acc_t O_acc[PV_N_TILES][4];
    #pragma unroll
    for (uint32_t n = 0; n < PV_N_TILES; ++n) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) O_acc[n][i] = (o_acc_t)0;
    }
    __syncthreads();

    auto issue_K = [&](uint32_t t0_target) {
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;
        constexpr uint32_t TOKENS_PER_PASS = (NWARPS * WARP_SIZE) / LANES_PER_ROW;
        constexpr uint32_t N_PASS = BC / TOKENS_PER_PASS;
        const uint32_t row_in_pass = tid / LANES_PER_ROW;
        const uint32_t col_lane    = tid % LANES_PER_ROW;
        #pragma unroll
        for (uint32_t pass = 0; pass < N_PASS; ++pass) {
            const uint32_t shm_t = pass * TOKENS_PER_PASS + row_in_pass;
            const uint32_t t     = t0_target + shm_t;
            const uint32_t base_d = col_lane * HALVES_PER_LD;
            __half *dst = &s_K[shm_t * HEAD_STRIDE + base_d];
            const bool in_range = (t < block_max_kv);
            if (USE_CP_ASYNC_K) {
                if (in_range) {
                    const __half *k_src = k_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    mma_detail::cp_async_cg_16(dst, k_src);
                } else {
                    mma_detail::cp_async_zero_16(dst);
                }
            } else {
                int4 *dst4 = reinterpret_cast<int4 *>(dst);
                if (in_range) {
                    const __half *k_src = k_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    *dst4 = *reinterpret_cast<const int4 *>(k_src);
                } else {
                    int4 z; z.x = z.y = z.z = z.w = 0;
                    *dst4 = z;
                }
            }
        }
    };

    auto issue_V = [&](uint32_t t0_target) {
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;
        constexpr uint32_t TOKENS_PER_PASS = (NWARPS * WARP_SIZE) / LANES_PER_ROW;
        constexpr uint32_t N_PASS = BC / TOKENS_PER_PASS;
        const uint32_t row_in_pass = tid / LANES_PER_ROW;
        const uint32_t col_lane    = tid % LANES_PER_ROW;
        #pragma unroll
        for (uint32_t pass = 0; pass < N_PASS; ++pass) {
            const uint32_t shm_t = pass * TOKENS_PER_PASS + row_in_pass;
            const uint32_t t     = t0_target + shm_t;
            const uint32_t base_d = col_lane * HALVES_PER_LD;
            const bool in_range = (t < block_max_kv);
            __half *dst = &s_V_buf[shm_t * HEAD_STRIDE + base_d];
            if (USE_CP_ASYNC_V) {
                if (in_range) {
                    const __half *v_src = v_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    mma_detail::cp_async_cg_16(dst, v_src);
                } else {
                    mma_detail::cp_async_zero_16(dst);
                }
            } else {
                __half v8[HALVES_PER_LD];
                if (in_range) {
                    const __half *v_src = v_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    *reinterpret_cast<int4 *>(v8) = *reinterpret_cast<const int4 *>(v_src);
                } else {
                    *reinterpret_cast<int4 *>(v8) = make_int4(0,0,0,0);
                }
                int4 *dst4 = reinterpret_cast<int4 *>(dst);
                *dst4 = *reinterpret_cast<int4 *>(v8);
            }
        }
    };

    // Prime: K[kv_lo].
    if (USE_CP_ASYNC_K) { issue_K(kv_lo); mma_detail::cp_async_commit(); }
    if (USE_CP_ASYNC_V) { issue_V(kv_lo); mma_detail::cp_async_commit(); }

    // Skip-out for entirely zero-padded m-tile (heads 6,7). With M_TILES=4
    // and active heads 0..5, mt=3's warp pair handles head_a=6 (active=false)
    // and head_b=7 (active=false). They still cooperate on K/V loads (all
    // warps must issue), but they skip the QK / softmax / PV work and write
    // zeros at writeback time. We collapse this by simply NOT executing
    // the QK accumulator update when both heads inactive — c_h stays zero,
    // and -INFINITY masking in score-write keeps softmax produce 0 P-values.
    // O_acc stays zero throughout.

    for (uint32_t t0 = kv_lo; t0 < kv_hi; t0 += BC) {
        if (USE_CP_ASYNC_K && USE_CP_ASYNC_V) {
            mma_detail::cp_async_wait_group<1>();
        } else if (USE_CP_ASYNC_K) {
            mma_detail::cp_async_wait_group<0>();
        }
        if (!USE_CP_ASYNC_K) issue_K(t0);
        if (!USE_CP_ASYNC_V) issue_V(t0);
        __syncthreads();

        // ---- QK^T phase. This warp does its m-tile × its BC half (2 nq tiles).
        float c_h[QK_N_TILES_PER_WARP][4];
        #pragma unroll
        for (uint32_t nq = 0; nq < QK_N_TILES_PER_WARP; ++nq) {
            c_h[nq][0]=0.f; c_h[nq][1]=0.f; c_h[nq][2]=0.f; c_h[nq][3]=0.f;
        }
        #pragma unroll
        for (uint32_t ks = 0; ks < QK_KSTEPS; ++ks) {
            const uint32_t k0 = ks * MMA_K;
            #pragma unroll
            for (uint32_t nq = 0; nq < QK_N_TILES_PER_WARP; ++nq) {
                const uint32_t kv_col = my_bc_lo + nq * MMA_N + group_id;
                const __half *krow0 = &s_K[kv_col * HEAD_STRIDE + k0 + in_group * 2];
                const unsigned b0 = mma_detail::pack2h(krow0);
                const unsigned b1 = mma_detail::pack2h(krow0 + 8);
                mma_detail::mma_m16n8k16_f16f16f32(
                    c_h[nq][0], c_h[nq][1], c_h[nq][2], c_h[nq][3],
                    Q_reg[ks][0], Q_reg[ks][1], Q_reg[ks][2], Q_reg[ks][3],
                    b0, b1);
            }
        }

        if (USE_CP_ASYNC_K) {
            const uint32_t t0_next = t0 + BC;
            if (t0_next < kv_hi) { issue_K(t0_next); mma_detail::cp_async_commit(); }
        }

        // ---- Write S to shmem with causal mask + head-padding mask.
        {
            const uint32_t q_idx_a = block_q * BR + r_a;
            const uint32_t q_idx_b = block_q * BR + r_b;
            const bool a_active = head_a_active && (q_idx_a < batch);
            const bool b_active = head_b_active && (q_idx_b < batch);
            const uint32_t my_max_kv_a = base_seq_len + q_idx_a + 1;
            const uint32_t my_max_kv_b = base_seq_len + q_idx_b + 1;
            const uint32_t m_a = mt * MMA_M + C_A * Q_ROWS_PER_TILE + r_a;
            const uint32_t m_b = mt * MMA_M + C_B * Q_ROWS_PER_TILE + r_b;
            #pragma unroll
            for (uint32_t nq = 0; nq < QK_N_TILES_PER_WARP; ++nq) {
                const uint32_t base_col = my_bc_lo + nq * MMA_N + in_group * 2;
                const uint32_t t_a0 = t0 + base_col;
                const uint32_t t_a1 = t0 + base_col + 1;
                float v00 = (a_active && t_a0 < my_max_kv_a) ? c_h[nq][0] * scale : -INFINITY;
                float v01 = (a_active && t_a1 < my_max_kv_a) ? c_h[nq][1] * scale : -INFINITY;
                float v02 = (b_active && t_a0 < my_max_kv_b) ? c_h[nq][2] * scale : -INFINITY;
                float v03 = (b_active && t_a1 < my_max_kv_b) ? c_h[nq][3] * scale : -INFINITY;
                s_S[m_a * S_STRIDE + base_col + 0] = v00;
                s_S[m_a * S_STRIDE + base_col + 1] = v01;
                s_S[m_b * S_STRIDE + base_col + 0] = v02;
                s_S[m_b * S_STRIDE + base_col + 1] = v03;
            }
        }
        __syncthreads();

        // ---- Online softmax over M_TOTAL=64 rows × BC=32 cols.
        // 8 warps × ROWS_PER_WARP=8 rows each. Each lane covers
        // BC/WARP_SIZE = 1 col.
        constexpr uint32_t ROWS_PER_WARP = M_TOTAL / NWARPS;       // 8
        constexpr uint32_t COLS_PER_LANE = BC / WARP_SIZE;         // 1
        #pragma unroll
        for (uint32_t r = 0; r < ROWS_PER_WARP; ++r) {
            const uint32_t m_row = warp * ROWS_PER_WARP + r;
            float local_max = -INFINITY;
            float s_vals[COLS_PER_LANE];
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                s_vals[cc] = s_S[m_row * S_STRIDE + lane * COLS_PER_LANE + cc];
                if (s_vals[cc] > local_max) local_max = s_vals[cc];
            }
            const float row_max = cuda_helpers::warp_reduce_max<32>(local_max);
            const float prev_m = s_m[m_row];
            const float new_m  = fmaxf(prev_m, row_max);
            float local_sum = 0.0f;
            __half pvals[COLS_PER_LANE];
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                const float p_val = (s_vals[cc] == -INFINITY) ? 0.0f
                                                              : __expf(s_vals[cc] - new_m);
                local_sum += p_val;
                pvals[cc] = __float2half(p_val);
            }
            const float row_sum = cuda_helpers::warp_reduce_sum<32>(local_sum);
            const float prev_l  = s_l[m_row];
            const float alpha   = (prev_m == -INFINITY) ? 0.0f : __expf(prev_m - new_m);
            const float new_l   = prev_l * alpha + row_sum;
            if (lane == 0) {
                s_m[m_row]     = new_m;
                s_l[m_row]     = new_l;
                s_alpha[m_row] = alpha;
            }
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                s_P[m_row * P_STRIDE + lane * COLS_PER_LANE + cc] = pvals[cc];
            }
        }

        if (USE_CP_ASYNC_V) {
            const bool k_next_pending = USE_CP_ASYNC_K && (t0 + BC < kv_hi);
            if (k_next_pending) mma_detail::cp_async_wait_group<1>();
            else                mma_detail::cp_async_wait_group<0>();
        }
        __syncthreads();

        // ---- Rescale O_acc by per-row alpha (this warp's m-tile only).
        const float alpha_a = s_alpha[mt * MMA_M + C_A * Q_ROWS_PER_TILE + group_id];
        const float alpha_b = s_alpha[mt * MMA_M + C_B * Q_ROWS_PER_TILE + group_id];
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            const float r0 = static_cast<float>(O_acc[n][0]) * alpha_a;
            const float r1 = static_cast<float>(O_acc[n][1]) * alpha_a;
            const float r2 = static_cast<float>(O_acc[n][2]) * alpha_b;
            const float r3 = static_cast<float>(O_acc[n][3]) * alpha_b;
            O_acc[n][0] = static_cast<o_acc_t>(r0);
            O_acc[n][1] = static_cast<o_acc_t>(r1);
            O_acc[n][2] = static_cast<o_acc_t>(r2);
            O_acc[n][3] = static_cast<o_acc_t>(r3);
        }

        // ---- PV phase. Each warp does ITS m-tile × ITS HD half.
        // P A-frag from s_P[mt*MMA_M..(mt+1)*MMA_M, k0..k0+16].
        // V B-frag from s_V_buf row-major; ldmatrix.x2.trans handles the n-axis.
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            const uint32_t n0 = warp_in_pair * HD_PER_WARP_IN_PAIR + n * MMA_N;
            #pragma unroll
            for (uint32_t ks = 0; ks < PV_KSTEPS; ++ks) {
                const uint32_t k0 = ks * MMA_K;
                unsigned b0, b1;
                if (USE_CP_ASYNC_V) {
                    const __half *vbase = &s_V_buf[k0 * HEAD_STRIDE + n0];
                    mma_detail::ldmatrix_x2_b_trans(b0, b1, vbase, HEAD_STRIDE);
                } else {
                    // Sync V load fallback (uses transposed layout).
                    const uint32_t out_col = n0 + group_id;
                    const __half *vrow0 = &s_V_buf[out_col * BC + k0 + in_group * 2];
                    b0 = mma_detail::pack2h(vrow0);
                    b1 = mma_detail::pack2h(vrow0 + 8);
                }
                unsigned a0, a1, a2, a3;
                mma_detail::ldmatrix_x4_a(a0, a1, a2, a3,
                                          &s_P[mt * MMA_M * P_STRIDE + k0], P_STRIDE);
                if constexpr (USE_FP16_O) {
                    float d0 = static_cast<float>(O_acc[n][0]);
                    float d1 = static_cast<float>(O_acc[n][1]);
                    float d2 = static_cast<float>(O_acc[n][2]);
                    float d3 = static_cast<float>(O_acc[n][3]);
                    mma_detail::mma_m16n8k16_f16f16f32(
                        d0, d1, d2, d3,
                        a0, a1, a2, a3, b0, b1);
                    O_acc[n][0] = static_cast<o_acc_t>(d0);
                    O_acc[n][1] = static_cast<o_acc_t>(d1);
                    O_acc[n][2] = static_cast<o_acc_t>(d2);
                    O_acc[n][3] = static_cast<o_acc_t>(d3);
                } else {
                    float &o0 = reinterpret_cast<float &>(O_acc[n][0]);
                    float &o1 = reinterpret_cast<float &>(O_acc[n][1]);
                    float &o2 = reinterpret_cast<float &>(O_acc[n][2]);
                    float &o3 = reinterpret_cast<float &>(O_acc[n][3]);
                    mma_detail::mma_m16n8k16_f16f16f32(
                        o0, o1, o2, o3,
                        a0, a1, a2, a3, b0, b1);
                }
            }
        }

        if (USE_CP_ASYNC_V) {
            const uint32_t t0_next = t0 + BC;
            if (t0_next < kv_hi) { issue_V(t0_next); mma_detail::cp_async_commit(); }
        }
    }

    // ---- Writeback / partial dump.
    const uint32_t q_idx_a = block_q * BR + r_a;
    const uint32_t q_idx_b = block_q * BR + r_b;
    const bool a_active = head_a_active && (q_idx_a < batch);
    const bool b_active = head_b_active && (q_idx_b < batch);
    const float l_a = s_l[mt * MMA_M + C_A * Q_ROWS_PER_TILE + group_id];
    const float l_b = s_l[mt * MMA_M + C_B * Q_ROWS_PER_TILE + group_id];
    const float m_a_val = s_m[mt * MMA_M + C_A * Q_ROWS_PER_TILE + group_id];
    const float m_b_val = s_m[mt * MMA_M + C_B * Q_ROWS_PER_TILE + group_id];

    if (NSPLIT == 1) {
        const float inv_la = (l_a > 0.0f) ? (1.0f / l_a) : 0.0f;
        const float inv_lb = (l_b > 0.0f) ? (1.0f / l_b) : 0.0f;
        float * const out_a = a_active
            ? (out + (uint64_t)q_idx_a * out_batch_stride + head_a * HEAD_DIM) : nullptr;
        float * const out_b = b_active
            ? (out + (uint64_t)q_idx_b * out_batch_stride + head_b * HEAD_DIM) : nullptr;
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            const uint32_t n0 = warp_in_pair * HD_PER_WARP_IN_PAIR + n * MMA_N;
            const uint32_t col = n0 + in_group * 2;
            if (out_a) {
                out_a[col + 0] = static_cast<float>(O_acc[n][0]) * inv_la;
                out_a[col + 1] = static_cast<float>(O_acc[n][1]) * inv_la;
            }
            if (out_b) {
                out_b[col + 0] = static_cast<float>(O_acc[n][2]) * inv_lb;
                out_b[col + 1] = static_cast<float>(O_acc[n][3]) * inv_lb;
            }
        }
    } else {
        // NSPLIT > 1: write un-normalized partials + (m, l) per (head, q_idx, k_split).
        const uint64_t pa_base = ((uint64_t)head_a * batch + q_idx_a) * NSPLIT + k_split;
        const uint64_t pb_base = ((uint64_t)head_b * batch + q_idx_b) * NSPLIT + k_split;
        if (warp_in_pair == 0 && in_group == 0) {
            if (a_active) {
                ms[pa_base * 2 + 0] = m_a_val;
                ms[pa_base * 2 + 1] = l_a;
            }
            if (b_active) {
                ms[pb_base * 2 + 0] = m_b_val;
                ms[pb_base * 2 + 1] = l_b;
            }
        }
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            const uint32_t n0 = warp_in_pair * HD_PER_WARP_IN_PAIR + n * MMA_N;
            const uint32_t col = n0 + in_group * 2;
            if (a_active) {
                partials[pa_base * HEAD_DIM + col + 0] = static_cast<float>(O_acc[n][0]);
                partials[pa_base * HEAD_DIM + col + 1] = static_cast<float>(O_acc[n][1]);
            }
            if (b_active) {
                partials[pb_base * HEAD_DIM + col + 0] = static_cast<float>(O_acc[n][2]);
                partials[pb_base * HEAD_DIM + col + 1] = static_cast<float>(O_acc[n][3]);
            }
        }
    }
}

static size_t fattn_mma_gqa_v4_smem_bytes(uint32_t head_dim) {
    constexpr uint32_t BR       = 8;
    constexpr uint32_t BC       = 32;
    constexpr uint32_t M_TOTAL  = 64;
    constexpr uint32_t KV_PAD   = 8;
    constexpr uint32_t S_PAD    = 4;
    const uint32_t hd_stride = head_dim + KV_PAD;
    const uint32_t s_stride  = BC + S_PAD;
    const uint32_t p_stride  = BC + S_PAD * 2;
    (void)BR;
    size_t s = 0;
    s += BC      * hd_stride * sizeof(__half);     // K
    s += BC      * hd_stride * sizeof(__half);     // V
    s += M_TOTAL * s_stride  * sizeof(float);      // S
    s += M_TOTAL * p_stride  * sizeof(__half);     // P
    s += M_TOTAL             * sizeof(float) * 3;  // m, l, alpha
    return s;
}

// =====================================================================
// fattn_prefill_mma_gqa_kernel_v5_t — BR=32 NCOLS2=2 + O_acc → s_O fp16
// =====================================================================
//
// q-rows-per-CTA = M_TOTAL = 64 (same as v3/v4) but with v2's lane mapping
// preserved (NCOLS2=2 q-head pack, M_TILES=4 stacked along m-axis). The
// register-pressure relief lever vs v2 BR=32: O accumulator moves from
// register array O_acc[M_TILES=4][PV_N_TILES=8][4] (= 128 fp32 = 128 regs/lane)
// to shared memory s_O[M_TOTAL=64][HD+OPAD=264] (= 33 KB, fp16). During the
// PV inner loop, only the per-(n) accumulator strip lives in regs:
// o_regs[M_TILES][4] = 16 fp32 = 16 regs/lane. Register pressure drops
// 128 → 16 for O, leaving budget for Q_reg[M_TILES=4][QK_KSTEPS=16][4]
// without the 4-KB BR=32 spill that v2 hits at the 255-reg cap.
//
// Tradeoffs:
//   - Shmem: 80.75 KB (vs v2 BR=16: 35 KB). 1 block/SM (vs v2: 2 blocks/SM).
//   - PV per (n) iter: 4 fp16 reads + 4 fp16 writes per (mt, lane) of s_O.
//     Folded with the rescale: read O_old fp16, multiply by alpha_a/alpha_b
//     into o_regs (eliminates v2's separate rescale phase), accumulate
//     PV_KSTEPS=2 MMAs in regs, write back fp16. Net round-trip density:
//     1 fp16↔fp32 round-trip per output element per tile (vs v2's 0;
//     vs v2 USE_FP16_O's PV_KSTEPS=2 round-trips per output per tile).
//   - Bank conflicts: O_PAD=8 halves on row stride breaks 8-way LDS
//     conflict on group_id read pattern (same insight as KV_PAD=8 / S_PAD=4).
//
// FP16 round-trip parity risk: lower than v2 USE_FP16_O standalone because
// the round-trip happens *outside* the ks loop (at the (mt, n) granularity),
// halving the demote/promote density per output. Tested: see
// feedback_fa2_v5_*.md.
//
// Shmem budget at BR=32, BC=32, KV_PAD=8, S_PAD=4, O_PAD=8:
//   K        : 32 × 264 × 2 = 16.5 KB
//   V        : 32 × 264 × 2 = 16.5 KB
//   S (fp32) : 64 × 36  × 4 =  9.0 KB
//   P (fp16) : 64 × 40  × 2 =  5.0 KB
//   O (fp16) : 64 × 264 × 2 = 33.0 KB
//   m,l,a    : 64 × 12      =  0.75 KB
//   total    :              ~ 80.75 KB  (1 block/SM @ 99 KB optin)
template <uint32_t HEAD_DIM, uint32_t Q_PER_KV, uint32_t BR, uint32_t BC,
          uint32_t NCOLS2, uint32_t KV_PAD, uint32_t S_PAD, uint32_t O_PAD,
          bool USE_CP_ASYNC_K, bool USE_CP_ASYNC_V, uint32_t NSPLIT>
__global__ void
__launch_bounds__(128, 1)
fattn_prefill_mma_gqa_kernel_v5_t(
        float       * __restrict__ out,
        float       * __restrict__ partials,
        float       * __restrict__ ms,
        const float * __restrict__ q,
        uint32_t     q_stride,
        const __half* __restrict__ k_cache,
        const __half* __restrict__ v_cache,
        uint32_t     n_heads,
        uint32_t     n_kv_heads,
        uint32_t     base_seq_len,
        uint32_t     batch,
        uint32_t     q_batch_stride,
        uint32_t     out_batch_stride,
        float        scale) {
    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint32_t NWARPS    = 4;
    constexpr uint32_t MMA_M     = 16;
    constexpr uint32_t MMA_N     = 8;
    constexpr uint32_t MMA_K     = 16;

    static_assert(BR == 16 || BR == 32, "v5 supports BR ∈ {16, 32}");
    static_assert(BC == 32,              "v5 supports BC=32");
    static_assert(NCOLS2 == 2,           "v5 supports NCOLS2=2");
    static_assert(KV_PAD == 8,           "v5 expects KV_PAD=8");
    static_assert(S_PAD  == 4,           "v5 expects S_PAD=4 fp32 elems");
    static_assert(O_PAD  == 8,           "v5 expects O_PAD=8 half elems");
    static_assert(BR % 8 == 0,           "BR must be multiple of 8");
    static_assert(Q_PER_KV % NCOLS2 == 0,"Q_PER_KV must divide by NCOLS2");
    static_assert(HEAD_DIM == 128 || HEAD_DIM == 256, "v5 supports HD 128/256");
    static_assert(HEAD_DIM % MMA_K == 0,"HEAD_DIM multiple of MMA_K");
    static_assert(BC % MMA_N == 0,      "BC multiple of MMA_N");
    static_assert(BC % NWARPS == 0,     "BC divisible by NWARPS");

    constexpr uint32_t S_STRIDE = BC + S_PAD;          // fp32 elements
    constexpr uint32_t P_STRIDE = BC + S_PAD * 2;      // half elements
    constexpr uint32_t O_STRIDE = HEAD_DIM + O_PAD;    // half elements

    constexpr uint32_t QK_KSTEPS  = HEAD_DIM / MMA_K;          // 16
    constexpr uint32_t QK_N_TILES = (BC / NWARPS) / MMA_N;     // 2
    constexpr uint32_t PV_KSTEPS  = BC / MMA_K;                // 2
    constexpr uint32_t HD_PER_WARP = HEAD_DIM / NWARPS;        // 64
    constexpr uint32_t PV_N_TILES  = HD_PER_WARP / MMA_N;      // 8
    constexpr uint32_t Q_GROUPS_PER_KV = Q_PER_KV / NCOLS2;    // 3
    constexpr uint32_t Q_ROWS_PER_TILE = MMA_M / NCOLS2;       // 8
    constexpr uint32_t M_TILES         = BR / Q_ROWS_PER_TILE; // 2@BR=16, 4@BR=32
    constexpr uint32_t M_TOTAL         = BR * NCOLS2;          // 32@BR=16, 64@BR=32
    static_assert(M_TILES * MMA_M == M_TOTAL, "M_TILES*MMA_M==M_TOTAL");
    static_assert(M_TOTAL % NWARPS == 0,      "M_TOTAL divisible by NWARPS");

    constexpr uint32_t HEAD_STRIDE = HEAD_DIM + KV_PAD;

    const uint32_t kvg_idx  = blockIdx.x;
    const uint32_t kv_head  = kvg_idx / Q_GROUPS_PER_KV;
    const uint32_t q_group  = kvg_idx % Q_GROUPS_PER_KV;
    const uint32_t block_q  = blockIdx.y;
    const uint32_t k_split  = blockIdx.z;
    const uint32_t tid      = threadIdx.x;
    const uint32_t warp     = tid / WARP_SIZE;
    const uint32_t lane     = tid % WARP_SIZE;
    if (kv_head >= n_kv_heads) return;
    const uint32_t head_base = kv_head * Q_PER_KV + q_group * NCOLS2;

    const uint32_t group_id = lane / 4;
    const uint32_t in_group = lane % 4;

    constexpr uint32_t C_A = 0;
    constexpr uint32_t C_B = 1;
    const uint32_t r_a = group_id;
    const uint32_t r_b = group_id;

    uint32_t block_max_q = block_q * BR + BR;
    if (block_max_q > batch) block_max_q = batch;
    const uint32_t block_max_kv = base_seq_len + block_max_q;

    uint32_t kv_lo, kv_hi;
    if (NSPLIT == 1) {
        kv_lo = 0;
        kv_hi = block_max_kv;
    } else {
        const uint32_t kv_total = base_seq_len + batch;
        const uint32_t per_split = ((kv_total + NSPLIT - 1) / NSPLIT + BC - 1) & ~uint32_t(BC - 1);
        kv_lo = k_split * per_split;
        kv_hi = kv_lo + per_split;
        if (kv_hi > block_max_kv) kv_hi = block_max_kv;
        if (kv_lo >= kv_hi) {
            #pragma unroll
            for (uint32_t mt = 0; mt < M_TILES; ++mt) {
                const uint32_t q_idx_a = block_q * BR + mt * Q_ROWS_PER_TILE + r_a;
                const uint32_t q_idx_b = block_q * BR + mt * Q_ROWS_PER_TILE + r_b;
                const uint32_t head_a  = head_base + C_A;
                const uint32_t head_b  = head_base + C_B;
                if (warp == 0 && in_group == 0) {
                    if (q_idx_a < batch) {
                        const uint64_t mb = ((uint64_t)head_a * batch + q_idx_a) * NSPLIT + k_split;
                        ms[mb * 2 + 0] = -INFINITY;
                        ms[mb * 2 + 1] = 0.0f;
                    }
                    if (q_idx_b < batch) {
                        const uint64_t mb = ((uint64_t)head_b * batch + q_idx_b) * NSPLIT + k_split;
                        ms[mb * 2 + 0] = -INFINITY;
                        ms[mb * 2 + 1] = 0.0f;
                    }
                }
                #pragma unroll
                for (uint32_t n = 0; n < PV_N_TILES; ++n) {
                    const uint32_t n0 = warp * HD_PER_WARP + n * MMA_N;
                    const uint32_t col = n0 + in_group * 2;
                    if (q_idx_a < batch) {
                        const uint64_t pbase = (((uint64_t)head_a * batch + q_idx_a)
                                                * NSPLIT + k_split) * HEAD_DIM;
                        partials[pbase + col + 0] = 0.0f;
                        partials[pbase + col + 1] = 0.0f;
                    }
                    if (q_idx_b < batch) {
                        const uint64_t pbase = (((uint64_t)head_b * batch + q_idx_b)
                                                * NSPLIT + k_split) * HEAD_DIM;
                        partials[pbase + col + 0] = 0.0f;
                        partials[pbase + col + 1] = 0.0f;
                    }
                }
            }
            return;
        }
    }

    extern __shared__ char smem_raw[];
    char *p = smem_raw;
    __half *s_K     = reinterpret_cast<__half *>(p); p += BC * HEAD_STRIDE * sizeof(__half);
    __half *s_V_buf = reinterpret_cast<__half *>(p); p += BC * HEAD_STRIDE * sizeof(__half);
    float  *s_S     = reinterpret_cast<float  *>(p); p += M_TOTAL * S_STRIDE * sizeof(float);
    __half *s_P     = reinterpret_cast<__half *>(p); p += M_TOTAL * P_STRIDE * sizeof(__half);
    __half *s_O     = reinterpret_cast<__half *>(p); p += M_TOTAL * O_STRIDE * sizeof(__half);
    float  *s_m     = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);
    float  *s_l     = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);
    float  *s_alpha = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);

    // ---- Q in registers (same packing as v2) -----------------------------
    unsigned Q_reg[M_TILES][QK_KSTEPS][4];
    {
        const uint32_t head_a = head_base + C_A;
        const uint32_t head_b = head_base + C_B;
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            const uint32_t q_idx_a = block_q * BR + mt * Q_ROWS_PER_TILE + r_a;
            const uint32_t q_idx_b = block_q * BR + mt * Q_ROWS_PER_TILE + r_b;
            const bool a_active = (q_idx_a < batch);
            const bool b_active = (q_idx_b < batch);
            const float *qa = a_active
                ? (q + (uint64_t)q_idx_a * q_batch_stride + head_a * q_stride) : nullptr;
            const float *qb = b_active
                ? (q + (uint64_t)q_idx_b * q_batch_stride + head_b * q_stride) : nullptr;
            #pragma unroll
            for (uint32_t ks = 0; ks < QK_KSTEPS; ++ks) {
                const uint32_t k0 = ks * MMA_K;
                const uint32_t col0 = k0 + in_group * 2;
                const __half qa0 = qa ? __float2half(qa[col0 + 0]) : (__half)0;
                const __half qa1 = qa ? __float2half(qa[col0 + 1]) : (__half)0;
                const __half qb0 = qb ? __float2half(qb[col0 + 0]) : (__half)0;
                const __half qb1 = qb ? __float2half(qb[col0 + 1]) : (__half)0;
                const __half qa8 = qa ? __float2half(qa[col0 + 8]) : (__half)0;
                const __half qa9 = qa ? __float2half(qa[col0 + 9]) : (__half)0;
                const __half qb8 = qb ? __float2half(qb[col0 + 8]) : (__half)0;
                const __half qb9 = qb ? __float2half(qb[col0 + 9]) : (__half)0;
                Q_reg[mt][ks][0] = mma_detail::pack2h(qa0, qa1);
                Q_reg[mt][ks][1] = mma_detail::pack2h(qb0, qb1);
                Q_reg[mt][ks][2] = mma_detail::pack2h(qa8, qa9);
                Q_reg[mt][ks][3] = mma_detail::pack2h(qb8, qb9);
            }
        }
    }

    if (tid < M_TOTAL) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }
    // Initialize s_O = 0 so the first-tile rescale (alpha=0) reads zeros
    // instead of uninitialized halves (NaN safety: 0 * NaN = NaN).
    {
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t TOTAL_HALVES  = M_TOTAL * O_STRIDE;
        constexpr uint32_t STEPS_PER_THR = (TOTAL_HALVES + 128 * HALVES_PER_LD - 1)
                                         / (128 * HALVES_PER_LD);
        const int4 zero4 = make_int4(0, 0, 0, 0);
        #pragma unroll
        for (uint32_t step = 0; step < STEPS_PER_THR; ++step) {
            const uint32_t off = (step * 128 + tid) * HALVES_PER_LD;
            if (off + HALVES_PER_LD <= TOTAL_HALVES) {
                *reinterpret_cast<int4 *>(&s_O[off]) = zero4;
            } else {
                #pragma unroll
                for (uint32_t i = 0; i < HALVES_PER_LD; ++i) {
                    if (off + i < TOTAL_HALVES) s_O[off + i] = (__half)0;
                }
            }
        }
    }
    __syncthreads();

    auto issue_K = [&](uint32_t t0_target) {
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;
        constexpr uint32_t TOKENS_PER_PASS = 128 / LANES_PER_ROW;
        constexpr uint32_t N_PASS = BC / TOKENS_PER_PASS;
        const uint32_t row_in_pass = tid / LANES_PER_ROW;
        const uint32_t col_lane    = tid % LANES_PER_ROW;
        #pragma unroll
        for (uint32_t pass = 0; pass < N_PASS; ++pass) {
            const uint32_t shm_t = pass * TOKENS_PER_PASS + row_in_pass;
            const uint32_t t     = t0_target + shm_t;
            const uint32_t base_d = col_lane * HALVES_PER_LD;
            __half *dst = &s_K[shm_t * HEAD_STRIDE + base_d];
            const bool in_range = (t < block_max_kv);
            if (USE_CP_ASYNC_K) {
                if (in_range) {
                    const __half *k_src = k_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    mma_detail::cp_async_cg_16(dst, k_src);
                } else {
                    mma_detail::cp_async_zero_16(dst);
                }
            } else {
                int4 *dst4 = reinterpret_cast<int4 *>(dst);
                if (in_range) {
                    const __half *k_src = k_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    *dst4 = *reinterpret_cast<const int4 *>(k_src);
                } else {
                    int4 z; z.x = z.y = z.z = z.w = 0;
                    *dst4 = z;
                }
            }
        }
    };

    auto issue_V = [&](uint32_t t0_target) {
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;
        constexpr uint32_t TOKENS_PER_PASS = 128 / LANES_PER_ROW;
        constexpr uint32_t N_PASS = BC / TOKENS_PER_PASS;
        const uint32_t row_in_pass = tid / LANES_PER_ROW;
        const uint32_t col_lane    = tid % LANES_PER_ROW;
        #pragma unroll
        for (uint32_t pass = 0; pass < N_PASS; ++pass) {
            const uint32_t shm_t = pass * TOKENS_PER_PASS + row_in_pass;
            const uint32_t t     = t0_target + shm_t;
            const uint32_t base_d = col_lane * HALVES_PER_LD;
            const bool in_range = (t < block_max_kv);
            if (USE_CP_ASYNC_V) {
                __half *dst = &s_V_buf[shm_t * HEAD_STRIDE + base_d];
                if (in_range) {
                    const __half *v_src = v_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    mma_detail::cp_async_cg_16(dst, v_src);
                } else {
                    mma_detail::cp_async_zero_16(dst);
                }
            } else {
                __half v8[HALVES_PER_LD];
                if (in_range) {
                    const __half *v_src = v_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    *reinterpret_cast<int4 *>(v8) = *reinterpret_cast<const int4 *>(v_src);
                } else {
                    *reinterpret_cast<int4 *>(v8) = make_int4(0,0,0,0);
                }
                #pragma unroll
                for (uint32_t i = 0; i < HALVES_PER_LD; ++i) {
                    s_V_buf[(base_d + i) * BC + shm_t] = v8[i];
                }
            }
        }
    };

    if (USE_CP_ASYNC_K) { issue_K(kv_lo); mma_detail::cp_async_commit(); }
    if (USE_CP_ASYNC_V) { issue_V(kv_lo); mma_detail::cp_async_commit(); }

    // ---- Tile loop -------------------------------------------------------
    for (uint32_t t0 = kv_lo; t0 < kv_hi; t0 += BC) {

        if (USE_CP_ASYNC_K && USE_CP_ASYNC_V) {
            mma_detail::cp_async_wait_group<1>();
        } else if (USE_CP_ASYNC_K) {
            mma_detail::cp_async_wait_group<0>();
        }
        if (!USE_CP_ASYNC_K) issue_K(t0);
        if (!USE_CP_ASYNC_V) issue_V(t0);
        __syncthreads();

        // QK^T phase. Same as v2.
        float c_h[M_TILES][QK_N_TILES][4];
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            #pragma unroll
            for (uint32_t nq = 0; nq < QK_N_TILES; ++nq) {
                c_h[mt][nq][0]=0.f; c_h[mt][nq][1]=0.f;
                c_h[mt][nq][2]=0.f; c_h[mt][nq][3]=0.f;
            }
        }
        #pragma unroll
        for (uint32_t ks = 0; ks < QK_KSTEPS; ++ks) {
            const uint32_t k0 = ks * MMA_K;
            #pragma unroll
            for (uint32_t nq = 0; nq < QK_N_TILES; ++nq) {
                const uint32_t kv_col = warp * (BC / NWARPS) + nq * MMA_N + group_id;
                const __half *krow0 = &s_K[kv_col * HEAD_STRIDE + k0 + in_group * 2];
                const unsigned b0 = mma_detail::pack2h(krow0);
                const unsigned b1 = mma_detail::pack2h(krow0 + 8);
                #pragma unroll
                for (uint32_t mt = 0; mt < M_TILES; ++mt) {
                    mma_detail::mma_m16n8k16_f16f16f32(
                        c_h[mt][nq][0], c_h[mt][nq][1], c_h[mt][nq][2], c_h[mt][nq][3],
                        Q_reg[mt][ks][0], Q_reg[mt][ks][1], Q_reg[mt][ks][2], Q_reg[mt][ks][3],
                        b0, b1);
                }
            }
        }

        if (USE_CP_ASYNC_K) {
            const uint32_t t0_next = t0 + BC;
            if (t0_next < kv_hi) { issue_K(t0_next); mma_detail::cp_async_commit(); }
        }

        // Write S to shmem with causal mask. Same lane mapping as v2.
        {
            #pragma unroll
            for (uint32_t mt = 0; mt < M_TILES; ++mt) {
                const uint32_t q_idx_a = block_q * BR + mt * Q_ROWS_PER_TILE + r_a;
                const uint32_t q_idx_b = block_q * BR + mt * Q_ROWS_PER_TILE + r_b;
                const bool a_active = (q_idx_a < batch);
                const bool b_active = (q_idx_b < batch);
                const uint32_t my_max_kv_a = base_seq_len + q_idx_a + 1;
                const uint32_t my_max_kv_b = base_seq_len + q_idx_b + 1;
                const uint32_t m_a = mt * MMA_M + C_A * Q_ROWS_PER_TILE + r_a;
                const uint32_t m_b = mt * MMA_M + C_B * Q_ROWS_PER_TILE + r_b;
                #pragma unroll
                for (uint32_t nq = 0; nq < QK_N_TILES; ++nq) {
                    const uint32_t base_col = warp * (BC / NWARPS) + nq * MMA_N + in_group * 2;
                    const uint32_t t_a0 = t0 + base_col;
                    const uint32_t t_a1 = t0 + base_col + 1;
                    float v00 = (a_active && t_a0 < my_max_kv_a) ? c_h[mt][nq][0] * scale : -INFINITY;
                    float v01 = (a_active && t_a1 < my_max_kv_a) ? c_h[mt][nq][1] * scale : -INFINITY;
                    float v02 = (b_active && t_a0 < my_max_kv_b) ? c_h[mt][nq][2] * scale : -INFINITY;
                    float v03 = (b_active && t_a1 < my_max_kv_b) ? c_h[mt][nq][3] * scale : -INFINITY;
                    s_S[m_a * S_STRIDE + base_col + 0] = v00;
                    s_S[m_a * S_STRIDE + base_col + 1] = v01;
                    s_S[m_b * S_STRIDE + base_col + 0] = v02;
                    s_S[m_b * S_STRIDE + base_col + 1] = v03;
                }
            }
        }

        __syncthreads();

        // Online softmax. Same as v2.
        constexpr uint32_t ROWS_PER_WARP = M_TOTAL / NWARPS;
        constexpr uint32_t COLS_PER_LANE = BC / WARP_SIZE;
        #pragma unroll
        for (uint32_t r = 0; r < ROWS_PER_WARP; ++r) {
            const uint32_t m_row = warp * ROWS_PER_WARP + r;
            float local_max = -INFINITY;
            float s_vals[COLS_PER_LANE];
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                s_vals[cc] = s_S[m_row * S_STRIDE + lane * COLS_PER_LANE + cc];
                if (s_vals[cc] > local_max) local_max = s_vals[cc];
            }
            const float row_max = cuda_helpers::warp_reduce_max<32>(local_max);
            const float prev_m = s_m[m_row];
            const float new_m  = fmaxf(prev_m, row_max);
            float local_sum = 0.0f;
            __half pvals[COLS_PER_LANE];
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                const float p_val = (s_vals[cc] == -INFINITY) ? 0.0f
                                                              : __expf(s_vals[cc] - new_m);
                local_sum += p_val;
                pvals[cc] = __float2half(p_val);
            }
            const float row_sum = cuda_helpers::warp_reduce_sum<32>(local_sum);
            const float prev_l  = s_l[m_row];
            const float alpha   = (prev_m == -INFINITY) ? 0.0f : __expf(prev_m - new_m);
            const float new_l   = prev_l * alpha + row_sum;
            if (lane == 0) {
                s_m[m_row]     = new_m;
                s_l[m_row]     = new_l;
                s_alpha[m_row] = alpha;
            }
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                s_P[m_row * P_STRIDE + lane * COLS_PER_LANE + cc] = pvals[cc];
            }
        }

        if (USE_CP_ASYNC_V) {
            const bool k_next_pending = USE_CP_ASYNC_K && (t0 + BC < kv_hi);
            if (k_next_pending) mma_detail::cp_async_wait_group<1>();
            else                mma_detail::cp_async_wait_group<0>();
        }
        __syncthreads();

        // Per-row alpha for rescale fold (same packing as v2).
        float alpha_a[M_TILES], alpha_b[M_TILES];
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            alpha_a[mt] = s_alpha[mt * MMA_M + group_id];
            alpha_b[mt] = s_alpha[mt * MMA_M + group_id + 8];
        }

        // ---- PV with O in shmem ------------------------------------------
        // Loop nest: for each n, for each mt, hold a 4-fp32 accumulator strip
        // in regs across PV_KSTEPS MMAs. At strip start, read fp16 O_old from
        // s_O and multiply by alpha (folds rescale into PV). At strip end,
        // demote to fp16 and write back to s_O.
        //
        // V is loaded M_TILES times more than v2 (mt is now inside n, ks).
        // PV_KSTEPS=2 here so V issue count = PV_N_TILES * PV_KSTEPS * M_TILES
        // = 8*2*4 = 64 ldmatrix.x2 per warp per tile (vs v2: 16). Cheap; the
        // win is the register-pressure relief (O_acc 128 → 16 regs/lane).
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            const uint32_t n0  = warp * HD_PER_WARP + n * MMA_N;
            const uint32_t col = n0 + in_group * 2;

            #pragma unroll
            for (uint32_t mt = 0; mt < M_TILES; ++mt) {
                const uint32_t m_a = mt * MMA_M + C_A * Q_ROWS_PER_TILE + r_a;
                const uint32_t m_b = mt * MMA_M + C_B * Q_ROWS_PER_TILE + r_b;

                // Read O_old (fp16) + rescale by alpha → 4 fp32 regs.
                const __half h0 = s_O[m_a * O_STRIDE + col + 0];
                const __half h1 = s_O[m_a * O_STRIDE + col + 1];
                const __half h2 = s_O[m_b * O_STRIDE + col + 0];
                const __half h3 = s_O[m_b * O_STRIDE + col + 1];
                float o0 = __half2float(h0) * alpha_a[mt];
                float o1 = __half2float(h1) * alpha_a[mt];
                float o2 = __half2float(h2) * alpha_b[mt];
                float o3 = __half2float(h3) * alpha_b[mt];

                #pragma unroll
                for (uint32_t ks = 0; ks < PV_KSTEPS; ++ks) {
                    const uint32_t k0 = ks * MMA_K;
                    unsigned b0, b1;
                    if (USE_CP_ASYNC_V) {
                        const __half *vbase = &s_V_buf[k0 * HEAD_STRIDE + n0];
                        mma_detail::ldmatrix_x2_b_trans(b0, b1, vbase, HEAD_STRIDE);
                    } else {
                        const uint32_t out_col = n0 + group_id;
                        const __half *vrow0 = &s_V_buf[out_col * BC + k0 + in_group * 2];
                        b0 = mma_detail::pack2h(vrow0);
                        b1 = mma_detail::pack2h(vrow0 + 8);
                    }
                    unsigned a0, a1, a2, a3;
                    mma_detail::ldmatrix_x4_a(a0, a1, a2, a3,
                                              &s_P[mt * MMA_M * P_STRIDE + k0], P_STRIDE);
                    mma_detail::mma_m16n8k16_f16f16f32(
                        o0, o1, o2, o3, a0, a1, a2, a3, b0, b1);
                }

                // Demote to fp16 and write back to s_O.
                s_O[m_a * O_STRIDE + col + 0] = __float2half(o0);
                s_O[m_a * O_STRIDE + col + 1] = __float2half(o1);
                s_O[m_b * O_STRIDE + col + 0] = __float2half(o2);
                s_O[m_b * O_STRIDE + col + 1] = __float2half(o3);
            }
        }

        if (USE_CP_ASYNC_V) {
            const uint32_t t0_next = t0 + BC;
            if (t0_next < kv_hi) { issue_V(t0_next); mma_detail::cp_async_commit(); }
        }
    }

    // ---- Final write -----------------------------------------------------
    // Each warp owns its own [n0, n0+HD_PER_WARP) cols of s_O, and each lane
    // wrote/reads exactly its own (m_a, m_b, col) cells, so no inter-thread
    // sync is needed before reading s_O for global write.
    {
        const uint32_t head_a = head_base + C_A;
        const uint32_t head_b = head_base + C_B;
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            const uint32_t q_idx_a = block_q * BR + mt * Q_ROWS_PER_TILE + r_a;
            const uint32_t q_idx_b = block_q * BR + mt * Q_ROWS_PER_TILE + r_b;
            const uint32_t m_a = mt * MMA_M + C_A * Q_ROWS_PER_TILE + r_a;
            const uint32_t m_b = mt * MMA_M + C_B * Q_ROWS_PER_TILE + r_b;
            const float l_a = s_l[m_a];
            const float l_b = s_l[m_b];
            if (NSPLIT == 1) {
                const float inv_la = (l_a > 0.0f) ? (1.0f / l_a) : 0.0f;
                const float inv_lb = (l_b > 0.0f) ? (1.0f / l_b) : 0.0f;
                float * const out_a = (q_idx_a < batch)
                    ? (out + (uint64_t)q_idx_a * out_batch_stride + head_a * HEAD_DIM) : nullptr;
                float * const out_b = (q_idx_b < batch)
                    ? (out + (uint64_t)q_idx_b * out_batch_stride + head_b * HEAD_DIM) : nullptr;
                #pragma unroll
                for (uint32_t n = 0; n < PV_N_TILES; ++n) {
                    const uint32_t n0  = warp * HD_PER_WARP + n * MMA_N;
                    const uint32_t col = n0 + in_group * 2;
                    if (out_a) {
                        out_a[col + 0] = __half2float(s_O[m_a * O_STRIDE + col + 0]) * inv_la;
                        out_a[col + 1] = __half2float(s_O[m_a * O_STRIDE + col + 1]) * inv_la;
                    }
                    if (out_b) {
                        out_b[col + 0] = __half2float(s_O[m_b * O_STRIDE + col + 0]) * inv_lb;
                        out_b[col + 1] = __half2float(s_O[m_b * O_STRIDE + col + 1]) * inv_lb;
                    }
                }
            } else {
                float * const part_a = (q_idx_a < batch)
                    ? (partials + (((uint64_t)head_a * batch + q_idx_a)
                                   * NSPLIT + k_split) * HEAD_DIM) : nullptr;
                float * const part_b = (q_idx_b < batch)
                    ? (partials + (((uint64_t)head_b * batch + q_idx_b)
                                   * NSPLIT + k_split) * HEAD_DIM) : nullptr;
                #pragma unroll
                for (uint32_t n = 0; n < PV_N_TILES; ++n) {
                    const uint32_t n0  = warp * HD_PER_WARP + n * MMA_N;
                    const uint32_t col = n0 + in_group * 2;
                    if (part_a) {
                        part_a[col + 0] = __half2float(s_O[m_a * O_STRIDE + col + 0]);
                        part_a[col + 1] = __half2float(s_O[m_a * O_STRIDE + col + 1]);
                    }
                    if (part_b) {
                        part_b[col + 0] = __half2float(s_O[m_b * O_STRIDE + col + 0]);
                        part_b[col + 1] = __half2float(s_O[m_b * O_STRIDE + col + 1]);
                    }
                }
                if (warp == 0 && in_group == 0) {
                    const float m_a_v = s_m[m_a];
                    const float m_b_v = s_m[m_b];
                    if (q_idx_a < batch) {
                        const uint64_t mb = ((uint64_t)head_a * batch + q_idx_a) * NSPLIT + k_split;
                        ms[mb * 2 + 0] = m_a_v;
                        ms[mb * 2 + 1] = l_a;
                    }
                    if (q_idx_b < batch) {
                        const uint64_t mb = ((uint64_t)head_b * batch + q_idx_b) * NSPLIT + k_split;
                        ms[mb * 2 + 0] = m_b_v;
                        ms[mb * 2 + 1] = l_b;
                    }
                }
            }
        }
    }
}

static size_t fattn_mma_gqa_v5_smem_bytes(uint32_t head_dim, uint32_t bc, uint32_t br) {
    constexpr uint32_t NCOLS2 = 2;
    constexpr uint32_t KV_PAD = 8;
    constexpr uint32_t S_PAD  = 4;
    constexpr uint32_t O_PAD  = 8;
    const uint32_t M_TOTAL    = br * NCOLS2;
    const uint32_t hd_stride  = head_dim + KV_PAD;
    const uint32_t s_stride   = bc + S_PAD;
    const uint32_t p_stride   = bc + S_PAD * 2;
    const uint32_t o_stride   = head_dim + O_PAD;
    size_t s = 0;
    s += bc      * hd_stride * sizeof(__half);   // K
    s += bc      * hd_stride * sizeof(__half);   // V
    s += M_TOTAL * s_stride  * sizeof(float);    // S
    s += M_TOTAL * p_stride  * sizeof(__half);   // P
    s += M_TOTAL * o_stride  * sizeof(__half);   // O (fp16, in shmem)
    s += M_TOTAL             * sizeof(float) * 3;// m, l, alpha
    return s;
}


// Combine kernel: merges NSPLIT partial (m, l, O_un_normalized) tuples per
// (head, q_token) via online softmax, divides by the merged denominator, and
// writes the final output.
//
// Layout:
//   partials [n_heads, batch, NSPLIT, HEAD_DIM]
//   ms       [n_heads, batch, NSPLIT, 2]   (col 0: m, col 1: l)
//   out      [batch, n_heads, HEAD_DIM]    (out_batch_stride between batches)
//
// Grid: (n_heads, batch). Block: HEAD_DIM threads — one thread per output dim.
template <uint32_t HEAD_DIM, uint32_t NSPLIT>
__global__ void
__launch_bounds__(HEAD_DIM, 2)
fattn_prefill_mma_gqa_combine_kernel(
        float       * __restrict__ out,
        const float * __restrict__ partials,
        const float * __restrict__ ms,
        uint32_t     batch,
        uint32_t     out_batch_stride) {
    const uint32_t head = blockIdx.x;
    const uint32_t b    = blockIdx.y;
    const uint32_t d    = threadIdx.x;
    if (d >= HEAD_DIM) return;

    const uint64_t mb_base = ((uint64_t)head * batch + b) * NSPLIT;

    __shared__ float s_max[NSPLIT];
    __shared__ float s_sum[NSPLIT];
    if (d < NSPLIT) {
        s_max[d] = ms[(mb_base + d) * 2 + 0];
        s_sum[d] = ms[(mb_base + d) * 2 + 1];
    }
    __syncthreads();

    // Global max across non-empty splits.
    float gmax = -INFINITY;
    #pragma unroll
    for (uint32_t s = 0; s < NSPLIT; ++s) {
        if (s_sum[s] > 0.0f && s_max[s] > gmax) gmax = s_max[s];
    }

    // Combine numerator and denominator.
    float gsum = 0.0f;
    float gnum = 0.0f;
    #pragma unroll
    for (uint32_t s = 0; s < NSPLIT; ++s) {
        if (s_sum[s] <= 0.0f) continue;
        const float w = __expf(s_max[s] - gmax);
        gsum += s_sum[s] * w;
        const uint64_t pbase = (mb_base + s) * HEAD_DIM;
        gnum += partials[pbase + d] * w;
    }

    float * out_ptr = out + static_cast<uint64_t>(b) * out_batch_stride;
    out_ptr[head * HEAD_DIM + d] = (gsum > 0.0f) ? (gnum / gsum) : 0.0f;
}

// Pick NSPLIT for the prefill GQA kernel.
//
// Negative result (2026-05-28, RTX Pro 6000, Qwen 3.6 27B):
//   Split-KV was hypothesized to close the long-T FA2 gap based on the
//   decode-side win (decode went 46% → 85% of llama.cpp at 128K with
//   NSPLIT=64). But the prefill base grid is `(n_kv_heads, n_blocks_q) =
//   (4, T/16)` — 16336 CTAs at T=64K — already 107 CTAs per SM in queue.
//   Splitting the K-axis 2× / 4× just makes 2× / 4× more CTAs of 1/2 / 1/4
//   work; total wall time is unchanged. Measured T=4K, 16K, 64K: NSPLIT=2
//   and NSPLIT=4 were within run-to-run noise of NSPLIT=1.
//
//   The long-T gap (17.65 s qw3 vs 4.71 s llama at 64K) must be intra-CTA
//   (per-tile __syncthreads count, register pressure, or lack of multi-
//   stage cp.async pipelining) — not parallelism. Split-KV stays as opt-in
//   infrastructure (parity-correct, allocates scratch) for future re-test
//   when other bottlenecks are addressed.
//
// Default: NSPLIT=1 (no scratch, no combine kernel).
// Override: QW3_PREFILL_FA2_NSPLIT=1/2/4 for diffs.
static inline uint32_t pick_prefill_gqa_nsplit(uint32_t batch, uint32_t n_kv_heads,
                                                uint32_t n_heads, uint32_t head_dim) {
    static const uint32_t kForced = []() -> uint32_t {
        const char *e = std::getenv("QW3_PREFILL_FA2_NSPLIT");
        if (!e) return 0;
        const int v = std::atoi(e);
        if (v == 1 || v == 2 || v == 4) return (uint32_t)v;
        return 0;
    }();
    if (kForced) return kForced;
    (void)batch; (void)n_kv_heads; (void)n_heads; (void)head_dim;
    return 1;
}

size_t fattn_prefill_mma_gqa_scratch_bytes(uint32_t batch, uint32_t n_heads,
                                            uint32_t n_kv_heads, uint32_t head_dim) {
    const uint32_t nsplit = pick_prefill_gqa_nsplit(batch, n_kv_heads, n_heads, head_dim);
    if (nsplit <= 1) return 0;
    const size_t partials = (size_t)n_heads * batch * nsplit * head_dim * sizeof(float);
    const size_t ms       = (size_t)n_heads * batch * nsplit * 2        * sizeof(float);
    return partials + ms;
}

// Pick NSPLIT for the v2 prefill kernel.
//
// Premise (verified 2026-05-30 on RTX Pro 6000, Qwen 3.6 27B 4 kv-heads):
//   v2's base grid = (n_kv_heads * 3, ceil(batch/BR), 1). At chunk=512 BR=16
//   the grid is 4*3*32 = 384 CTAs. With v2's 35 KB shmem allowing 2 blocks/SM,
//   the resident slot budget is 152*2 = 304. So one wave of 304 + a tail of 80 —
//   1.26 waves with the last wave 26%-occupied. Splitting K 2× / 4× scales the
//   grid to 768 / 1536 CTAs (2.5 / 5 waves), packing the SMs more densely.
//
//   At whole-prompt T (chunk=0), batch is the full prompt and the grid already
//   spans many waves; NSPLIT=1 stays correct. At chunk=512 long-T (the regime
//   where v2 trails llama by ~25-40%), bumping NSPLIT can stack on top of
//   v2's q-rows-per-CTA gain.
//
// Cut-off: split when q_blocks * kv_groups * Q_GROUPS_PER_KV ≤ 304 * 1.5 = 456.
//   chunk=512 BR=16: 32 * 4 * 3 = 384  → split (NSPLIT=2 brings to 768)
//   chunk=2048 BR=16: 128*4*3=1536      → no split
//   T=65K whole: ~4096*4*3 ≈ 49K        → no split
//
// NSPLIT=2 is the conservative pick. NSPLIT=4 was useful on v1 (NSPLIT=1
// underutilized), but on v2 the head-pack already halves the per-CTA grid:
// going from 1.26 waves to 5 waves (NSPLIT=4) over-shards each CTA's work.
// Stay at NSPLIT=2 unless QW3_PREFILL_FA2_NSPLIT forces otherwise.
static inline uint32_t pick_prefill_gqa_v2_nsplit(uint32_t batch, uint32_t n_kv_heads,
                                                   uint32_t n_heads, uint32_t head_dim,
                                                   uint32_t br) {
    static const uint32_t kForced = []() -> uint32_t {
        const char *e = std::getenv("QW3_PREFILL_FA2_NSPLIT");
        if (!e) return 0;
        const int v = std::atoi(e);
        if (v == 1 || v == 2 || v == 4) return (uint32_t)v;
        return 0;
    }();
    if (kForced) return kForced;
    (void)n_heads; (void)head_dim;
    constexpr uint32_t NCOLS2 = 2;
    const uint32_t q_per_kv      = 6;                          // gated check
    const uint32_t q_groups      = q_per_kv / NCOLS2;          // 3
    const uint32_t q_blocks      = (batch + br - 1) / br;
    const uint32_t base_grid     = q_blocks * n_kv_heads * q_groups;
    // 2 blocks/SM × 152 SMs = 304 resident slots; allow 1.5× headroom before
    // splitting (i.e. only split when the base grid leaves >50% of a wave idle).
    constexpr uint32_t kSatThreshold = 304u + 152u;            // 456
    return (base_grid <= kSatThreshold) ? 2u : 1u;
}

size_t fattn_prefill_mma_gqa_v2_scratch_bytes(uint32_t batch, uint32_t n_heads,
                                                uint32_t n_kv_heads, uint32_t head_dim) {
    static const uint32_t br_choice = []() -> uint32_t {
        const char *e = std::getenv("QW3_PREFILL_FA2_BR");
        if (!e) return 16;
        if (std::strcmp(e, "8")  == 0) return 8;
        if (std::strcmp(e, "16") == 0) return 16;
        if (std::strcmp(e, "32") == 0) return 32;
        return 16;
    }();
    const uint32_t nsplit = pick_prefill_gqa_v2_nsplit(batch, n_kv_heads, n_heads, head_dim, br_choice);
    if (nsplit <= 1) return 0;
    const size_t partials = (size_t)n_heads * batch * nsplit * head_dim * sizeof(float);
    const size_t ms       = (size_t)n_heads * batch * nsplit * 2        * sizeof(float);
    return partials + ms;
}

// Piped kernel: drops Q (in registers), adds a 2nd K buffer for cp.async ping-pong.
// HD=256: 43 - 8 (Q) + 16 (K[1]) = 51 KB ; HD=128: 24 - 4 + 8 = 28 KB.
static size_t fattn_mma_smem_bytes_pipe(uint32_t head_dim) {
    constexpr uint32_t BR = 16, BC = 32;
    size_t s = 0;
    s += 2 * BC   * head_dim * sizeof(__half);   // K[2] (also reused for Q init)
    s += head_dim * BC       * sizeof(__half);   // V_T
    s += BR       * BC       * sizeof(float);    // S
    s += BR       * BC       * sizeof(__half);   // P
    s += BR                  * sizeof(float) * 3;// m, l, alpha
    return s;
}

bool launch_fattn_prefill_mma_f16(
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
    if (!(head_dim == 128 || head_dim == 256)) return false;
    if (batch == 0) return true;
    constexpr uint32_t BR = 16, BC = 32;
    const uint32_t n_blocks_q = (batch + BR - 1) / BR;
    const dim3 grid(n_heads, n_blocks_q, 1);
    const dim3 block(128);
    const size_t smem = fattn_mma_smem_bytes(head_dim);
    auto launch = [&](auto HD_v) {
        constexpr uint32_t HD = decltype(HD_v)::value;
        cudaFuncSetAttribute(fattn_prefill_mma_kernel<HD, BR, BC>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        fattn_prefill_mma_kernel<HD, BR, BC><<<grid, block, smem, stream>>>(
            out, q, q_stride,
            static_cast<const __half *>(k_cache),
            static_cast<const __half *>(v_cache),
            n_heads, n_kv_heads, base_seq_len, batch,
            q_batch_stride, out_batch_stride, scale);
    };
    if (head_dim == 256) launch(std::integral_constant<uint32_t, 256>{});
    else                 launch(std::integral_constant<uint32_t, 128>{});
    return true;
}

bool launch_fattn_prefill_mma_pipe_f16(
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
    if (!(head_dim == 128 || head_dim == 256)) return false;
    if (batch == 0) return true;
    constexpr uint32_t BR = 16, BC = 32;
    const uint32_t n_blocks_q = (batch + BR - 1) / BR;
    const dim3 grid(n_heads, n_blocks_q, 1);
    const dim3 block(128);
    const size_t smem = fattn_mma_smem_bytes_pipe(head_dim);
    auto launch = [&](auto HD_v) {
        constexpr uint32_t HD = decltype(HD_v)::value;
        cudaFuncSetAttribute(fattn_prefill_mma_kernel_pipe<HD, BR, BC>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        fattn_prefill_mma_kernel_pipe<HD, BR, BC><<<grid, block, smem, stream>>>(
            out, q, q_stride,
            static_cast<const __half *>(k_cache),
            static_cast<const __half *>(v_cache),
            n_heads, n_kv_heads, base_seq_len, batch,
            q_batch_stride, out_batch_stride, scale);
    };
    if (head_dim == 256) launch(std::integral_constant<uint32_t, 256>{});
    else                 launch(std::integral_constant<uint32_t, 128>{});
    return true;
}

bool launch_fattn_prefill_mma_gqa_f16(
        float *       out,
        void  *       scratch,
        size_t        scratch_bytes,
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
    if (!(head_dim == 128 || head_dim == 256)) return false;
    if (batch == 0) return true;
    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return false;
    const uint32_t q_per_kv = n_heads / n_kv_heads;
    if (!(q_per_kv == 6)) return false;   // Qwen 3.6 27B GQA group size
    constexpr uint32_t BR = 16, BC = 32;
    const uint32_t n_blocks_q = (batch + BR - 1) / BR;
    const size_t smem = fattn_mma_gqa_smem_bytes(head_dim, q_per_kv);
    // QW3_PREFILL_FA2_KCPASYNC: cp.async double-buffer the K tile load. The
    // K[t+1] gmem read is issued post-QK (when s_K is dead) and overlaps the
    // current tile's softmax + PV. Default ON — measured +5-7% at long T,
    // ±0.5% noise at short T on Qwen 3.6 27B / RTX Pro 6000. Set to "0" to
    // force the synchronous K-load fallback.
    static const bool use_cp_async_k = []() {
        const char *e = std::getenv("QW3_PREFILL_FA2_KCPASYNC");
        if (!e) return true;
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0);
    }();
    // QW3_PREFILL_FA2_VCPASYNC: cp.async double-buffer the V tile load.
    // Requires the row-major s_V[t,d] shmem layout and ldmatrix.x2.trans on
    // the PV-side B operand. V[t+1] is issued post-PV (when s_V is dead)
    // and overlaps the next iter's K wait + Q init + softmax. Default ON.
    // Set to "0" to fall back to the legacy transposed sync V load.
    static const bool use_cp_async_v = []() {
        const char *e = std::getenv("QW3_PREFILL_FA2_VCPASYNC");
        if (!e) return true;
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0);
    }();

    // Pick NSPLIT (KV-axis cooperative reduction). Returns 1 when the base
    // grid already saturates SMs or scratch budget is tight; ≥2 when long-T
    // serial K-walk per CTA is the bottleneck.
    const uint32_t nsplit = pick_prefill_gqa_nsplit(batch, n_kv_heads, n_heads, head_dim);
    float *partials = nullptr, *ms = nullptr;
    if (nsplit > 1) {
        const size_t need = (size_t)n_heads * batch * nsplit * head_dim * sizeof(float)
                          + (size_t)n_heads * batch * nsplit * 2        * sizeof(float);
        if (!scratch || scratch_bytes < need) {
            // Scratch too small — caller didn't size it for split. Fall back
            // to NSPLIT=1 path.
            // (This shouldn't happen in normal use; the caller should query
            //  fattn_prefill_mma_gqa_scratch_bytes first.)
            return false;
        }
        partials = static_cast<float *>(scratch);
        ms       = partials + (size_t)n_heads * batch * nsplit * head_dim;
    }

    auto launch = [&](auto HD_v, auto NS_v) {
        constexpr uint32_t HD = decltype(HD_v)::value;
        constexpr uint32_t NS = decltype(NS_v)::value;
        const dim3 grid(n_kv_heads, n_blocks_q, NS);
        const dim3 block(128);
        auto dispatch = [&](auto K_v, auto V_v) {
            constexpr bool K = decltype(K_v)::value;
            constexpr bool V = decltype(V_v)::value;
            cudaFuncSetAttribute(
                fattn_prefill_mma_gqa_kernel_t<HD, 6, BR, BC, K, V, NS>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            fattn_prefill_mma_gqa_kernel_t<HD, 6, BR, BC, K, V, NS>
                <<<grid, block, smem, stream>>>(
                    out, partials, ms, q, q_stride,
                    static_cast<const __half *>(k_cache),
                    static_cast<const __half *>(v_cache),
                    n_heads, n_kv_heads, base_seq_len, batch,
                    q_batch_stride, out_batch_stride, scale);
        };
        if (use_cp_async_k && use_cp_async_v) {
            dispatch(std::true_type{},  std::true_type{});
        } else if (use_cp_async_k) {
            dispatch(std::true_type{},  std::false_type{});
        } else if (use_cp_async_v) {
            dispatch(std::false_type{}, std::true_type{});
        } else {
            dispatch(std::false_type{}, std::false_type{});
        }
        if (NS > 1) {
            const dim3 cgrid(n_heads, batch, 1);
            const dim3 cblock(HD);
            fattn_prefill_mma_gqa_combine_kernel<HD, NS>
                <<<cgrid, cblock, 0, stream>>>(
                    out, partials, ms, batch, out_batch_stride);
        }
    };

    auto launch_hd = [&](auto HD_v) {
        if      (nsplit == 4) launch(HD_v, std::integral_constant<uint32_t, 4>{});
        else if (nsplit == 2) launch(HD_v, std::integral_constant<uint32_t, 2>{});
        else                  launch(HD_v, std::integral_constant<uint32_t, 1>{});
    };
    if (head_dim == 256) launch_hd(std::integral_constant<uint32_t, 256>{});
    else                 launch_hd(std::integral_constant<uint32_t, 128>{});
    return true;
}

// v2 launcher: BR=8/16 + ncols2=2 + Q-in-regs. Optionally split-KV (NSPLIT
// chosen by SM-saturation heuristic in pick_prefill_gqa_v2_nsplit). When
// NSPLIT>1 the caller must size `scratch` per fattn_prefill_mma_gqa_v2_scratch_bytes;
// the launcher dispatches the combine kernel to reduce partials. Currently
// gated to Qwen 3.6 27B's q_per_kv=6.
bool launch_fattn_prefill_mma_gqa_v2_f16(
        float *       out,
        void  *       scratch,
        size_t        scratch_bytes,
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
    if (!(head_dim == 128 || head_dim == 256)) return false;
    if (batch == 0) return true;
    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return false;
    const uint32_t q_per_kv = n_heads / n_kv_heads;
    if (!(q_per_kv == 6)) return false;
    constexpr uint32_t NCOLS2 = 2;

    static const uint32_t bc_choice = []() -> uint32_t {
        const char *e = std::getenv("QW3_PREFILL_FA2_BC");
        if (!e) return 32;
        if (std::strcmp(e, "32") == 0) return 32;
        if (std::strcmp(e, "64") == 0) return 64;
        return 32;
    }();
    static const uint32_t br_choice = []() -> uint32_t {
        const char *e = std::getenv("QW3_PREFILL_FA2_BR");
        if (!e) return 16;
        if (std::strcmp(e, "8")  == 0) return 8;
        if (std::strcmp(e, "16") == 0) return 16;
        // BR=32 NCOLS2=2 (M_TILES=4) compiles but regresses 1-1.5% across T —
        // 4 KB Q_reg spill at 255-reg cap lands on hot path. Kept reachable
        // via env for diagnostic / future-rewrite use only.
        if (std::strcmp(e, "32") == 0) return 32;
        return 16;
    }();
    // K/V row-stride padding (in halves) to break the 8-way LDS bank conflict
    // on the QK pack2h read path. Default 0 (no pad). KV_PAD=8 shifts each
    // adjacent K row by 4 banks, mapping the 8 group_id lanes onto distinct
    // banks. +1 KB shmem at HD=256 BC=32; still 2 blocks/SM.
    static const uint32_t kv_pad_choice = []() -> uint32_t {
        const char *e = std::getenv("QW3_PREFILL_FA2_KPAD");
        if (!e) return 8;
        if (std::strcmp(e, "0") == 0) return 0;
        if (std::strcmp(e, "8") == 0) return 8;
        return 8;
    }();
    // s_S/s_P row-stride padding (in fp32 elements; bytes = pad*4 for both
    // since s_P is __half-typed and we use 2× the element pad to match bytes).
    // S_PAD=4 fp32 elems = 16 B = 4 banks. Maps adjacent score-tile rows onto
    // distinct bank quartets; targets the NCU-measured 45% L1 uncoalesced
    // shmem-wave count on v2's softmax + PV ldmatrix.x4 path. +0.5 KB shmem
    // at BR=16 BC=32; still 2 blocks/SM.
    static const uint32_t s_pad_choice = []() -> uint32_t {
        const char *e = std::getenv("QW3_PREFILL_FA2_SPAD");
        if (!e) return 4;
        if (std::strcmp(e, "0") == 0) return 0;
        if (std::strcmp(e, "4") == 0) return 4;
        return 4;
    }();

    const uint32_t n_blocks_q = (batch + br_choice - 1) / br_choice;
    const uint32_t q_groups   = q_per_kv / NCOLS2;     // 3
    const uint32_t nsplit = pick_prefill_gqa_v2_nsplit(batch, n_kv_heads, n_heads, head_dim, br_choice);
    float *partials = nullptr, *ms_buf = nullptr;
    if (nsplit > 1) {
        const size_t need = (size_t)n_heads * batch * nsplit * head_dim * sizeof(float)
                          + (size_t)n_heads * batch * nsplit * 2        * sizeof(float);
        if (!scratch || scratch_bytes < need) return false;
        partials = static_cast<float *>(scratch);
        ms_buf   = partials + (size_t)n_heads * batch * nsplit * head_dim;
    }
    const dim3 grid(n_kv_heads * q_groups, n_blocks_q, nsplit);
    const dim3 block(128);
    const size_t smem = fattn_mma_gqa_v2_smem_bytes(head_dim, bc_choice, br_choice, kv_pad_choice, s_pad_choice);

    static const bool use_cp_async_k = []() {
        const char *e = std::getenv("QW3_PREFILL_FA2_KCPASYNC");
        if (!e) return true;
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0);
    }();
    static const bool use_cp_async_v = []() {
        const char *e = std::getenv("QW3_PREFILL_FA2_VCPASYNC");
        if (!e) return true;
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0);
    }();
    // QW3_PREFILL_FA2_FP16_O: store O_acc in __half instead of float. MMA
    // stays f16f16f32; we promote FP16→FP32 around each MMA. Saves ~256 B/
    // thread of register pressure at BR=16, more at BR=32/64. Off by default
    // until parity at long T is confirmed.
    static const bool use_fp16_o = []() {
        const char *e = std::getenv("QW3_PREFILL_FA2_FP16_O");
        if (!e) return false;
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0);
    }();

    auto launch = [&](auto HD_v, auto BR_v, auto BC_v, auto KPAD_v, auto SPAD_v, auto NS_v) {
        constexpr uint32_t HD   = decltype(HD_v)::value;
        constexpr uint32_t BR   = decltype(BR_v)::value;
        constexpr uint32_t BC   = decltype(BC_v)::value;
        constexpr uint32_t KPAD = decltype(KPAD_v)::value;
        constexpr uint32_t SPAD = decltype(SPAD_v)::value;
        constexpr uint32_t NS   = decltype(NS_v)::value;
        auto dispatch = [&](auto K_v, auto V_v, auto FP16O_v) {
            constexpr bool K = decltype(K_v)::value;
            constexpr bool V = decltype(V_v)::value;
            constexpr bool FP16O = decltype(FP16O_v)::value;
            cudaFuncSetAttribute(
                fattn_prefill_mma_gqa_kernel_v2_t<HD, 6, BR, BC, NCOLS2, KPAD, K, V, NS, FP16O, SPAD>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            fattn_prefill_mma_gqa_kernel_v2_t<HD, 6, BR, BC, NCOLS2, KPAD, K, V, NS, FP16O, SPAD>
                <<<grid, block, smem, stream>>>(
                    out, partials, ms_buf, q, q_stride,
                    static_cast<const __half *>(k_cache),
                    static_cast<const __half *>(v_cache),
                    n_heads, n_kv_heads, base_seq_len, batch,
                    q_batch_stride, out_batch_stride, scale);
        };
        auto dispatch_fp16o = [&](auto K_v, auto V_v) {
            if (use_fp16_o) dispatch(K_v, V_v, std::true_type{});
            else            dispatch(K_v, V_v, std::false_type{});
        };
        if (use_cp_async_k && use_cp_async_v) {
            dispatch_fp16o(std::true_type{},  std::true_type{});
        } else if (use_cp_async_k) {
            dispatch_fp16o(std::true_type{},  std::false_type{});
        } else if (use_cp_async_v) {
            dispatch_fp16o(std::false_type{}, std::true_type{});
        } else {
            dispatch_fp16o(std::false_type{}, std::false_type{});
        }
        if (NS > 1) {
            const dim3 cgrid(n_heads, batch, 1);
            const dim3 cblock(HD);
            fattn_prefill_mma_gqa_combine_kernel<HD, NS>
                <<<cgrid, cblock, 0, stream>>>(
                    out, partials, ms_buf, batch, out_batch_stride);
        }
    };

    auto launch_with_ns = [&](auto HD_v, auto BR_v, auto BC_v, auto KPAD_v, auto SPAD_v) {
        if (nsplit == 2) launch(HD_v, BR_v, BC_v, KPAD_v, SPAD_v, std::integral_constant<uint32_t, 2>{});
        else             launch(HD_v, BR_v, BC_v, KPAD_v, SPAD_v, std::integral_constant<uint32_t, 1>{});
    };
    auto launch_with_spad = [&](auto HD_v, auto BR_v, auto BC_v, auto KPAD_v) {
        if (s_pad_choice == 4) launch_with_ns(HD_v, BR_v, BC_v, KPAD_v, std::integral_constant<uint32_t, 4>{});
        else                   launch_with_ns(HD_v, BR_v, BC_v, KPAD_v, std::integral_constant<uint32_t, 0>{});
    };
    auto launch_with_kpad = [&](auto HD_v, auto BR_v, auto BC_v) {
        if (kv_pad_choice == 8) launch_with_spad(HD_v, BR_v, BC_v, std::integral_constant<uint32_t, 8>{});
        else                    launch_with_spad(HD_v, BR_v, BC_v, std::integral_constant<uint32_t, 0>{});
    };
    auto launch_with_bc = [&](auto HD_v, auto BR_v) {
        if (bc_choice == 32) launch_with_kpad(HD_v, BR_v, std::integral_constant<uint32_t, 32>{});
        else                 launch_with_kpad(HD_v, BR_v, std::integral_constant<uint32_t, 64>{});
    };
    auto launch_with_br = [&](auto HD_v) {
        if      (br_choice == 32) launch_with_bc(HD_v, std::integral_constant<uint32_t, 32>{});
        else if (br_choice == 16) launch_with_bc(HD_v, std::integral_constant<uint32_t, 16>{});
        else                      launch_with_bc(HD_v, std::integral_constant<uint32_t, 8>{});
    };
    if (head_dim == 256) launch_with_br(std::integral_constant<uint32_t, 256>{});
    else                 launch_with_br(std::integral_constant<uint32_t, 128>{});
    return true;
}

// v3 launcher: BR=64 + NCOLS2=1 + Q-in-shmem. The q-rows-per-CTA lever
// applied: 64 q-rows/CTA, 4 MMAs per K/V load. Long-T attack.
// Currently gated to Qwen 3.6 27B's q_per_kv=6.
bool launch_fattn_prefill_mma_gqa_v3_f16(
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
    if (!(head_dim == 128 || head_dim == 256)) return false;
    if (batch == 0) return true;
    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return false;
    const uint32_t q_per_kv = n_heads / n_kv_heads;
    if (!(q_per_kv == 6)) return false;

    static const uint32_t bc_choice = []() -> uint32_t {
        const char *e = std::getenv("QW3_PREFILL_FA2_BC");
        if (!e) return 32;
        if (std::strcmp(e, "32") == 0) return 32;
        if (std::strcmp(e, "64") == 0) return 64;
        return 32;
    }();
    static const uint32_t br_choice = []() -> uint32_t {
        const char *e = std::getenv("QW3_PREFILL_FA2_BR");
        if (!e) return 64;
        if (std::strcmp(e, "32") == 0) return 32;
        if (std::strcmp(e, "64") == 0) return 64;
        return 64;
    }();

    const uint32_t n_blocks_q = (batch + br_choice - 1) / br_choice;
    // Grid: each q-head is its own CTA (NCOLS2=1).
    const dim3 grid(n_kv_heads * q_per_kv, n_blocks_q, 1);
    const dim3 block(128);
    const size_t smem = fattn_mma_gqa_v3_smem_bytes(head_dim, bc_choice, br_choice);

    static const bool use_cp_async_k = []() {
        const char *e = std::getenv("QW3_PREFILL_FA2_KCPASYNC");
        if (!e) return true;
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0);
    }();
    static const bool use_cp_async_v = []() {
        const char *e = std::getenv("QW3_PREFILL_FA2_VCPASYNC");
        if (!e) return true;
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0);
    }();

    auto launch = [&](auto HD_v, auto BR_v, auto BC_v) {
        constexpr uint32_t HD = decltype(HD_v)::value;
        constexpr uint32_t BR = decltype(BR_v)::value;
        constexpr uint32_t BC = decltype(BC_v)::value;
        auto dispatch = [&](auto K_v, auto V_v) {
            constexpr bool K = decltype(K_v)::value;
            constexpr bool V = decltype(V_v)::value;
            cudaFuncSetAttribute(
                fattn_prefill_mma_gqa_kernel_v3_t<HD, 6, BR, BC, K, V>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            fattn_prefill_mma_gqa_kernel_v3_t<HD, 6, BR, BC, K, V>
                <<<grid, block, smem, stream>>>(
                    out, q, q_stride,
                    static_cast<const __half *>(k_cache),
                    static_cast<const __half *>(v_cache),
                    n_heads, n_kv_heads, base_seq_len, batch,
                    q_batch_stride, out_batch_stride, scale);
        };
        if (use_cp_async_k && use_cp_async_v) {
            dispatch(std::true_type{},  std::true_type{});
        } else if (use_cp_async_k) {
            dispatch(std::true_type{},  std::false_type{});
        } else if (use_cp_async_v) {
            dispatch(std::false_type{}, std::true_type{});
        } else {
            dispatch(std::false_type{}, std::false_type{});
        }
    };

    auto launch_with_bc = [&](auto HD_v, auto BR_v) {
        if (bc_choice == 32) launch(HD_v, BR_v, std::integral_constant<uint32_t, 32>{});
        else                 launch(HD_v, BR_v, std::integral_constant<uint32_t, 64>{});
    };
    auto launch_with_br = [&](auto HD_v) {
        if (br_choice == 64) launch_with_bc(HD_v, std::integral_constant<uint32_t, 64>{});
        else                 launch_with_bc(HD_v, std::integral_constant<uint32_t, 32>{});
    };
    if (head_dim == 256) launch_with_br(std::integral_constant<uint32_t, 256>{});
    else                 launch_with_br(std::integral_constant<uint32_t, 128>{});
    return true;
}

// =====================================================================
// v4 launcher: NWARPS=8 + warp-pair-owns-mtile, M_TOTAL=64.
// =====================================================================
// Currently gated to Qwen 3.6 27B's q_per_kv=6 (kernel hard-codes Q_PER_KV=6
// with q-head zero-pad to 8).
size_t fattn_prefill_mma_gqa_v4_scratch_bytes(uint32_t batch, uint32_t n_heads,
                                              uint32_t n_kv_heads,
                                              uint32_t head_dim) {
    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return 0;
    const uint32_t q_per_kv = n_heads / n_kv_heads;
    if (q_per_kv != 6) return 0;
    if (!(head_dim == 128 || head_dim == 256)) return 0;
    // v4 reuses the same NSPLIT scratch shape as v2: per (head, q_tok, split)
    // we stash HD floats of un-normalized O plus 2 floats of (m, l).
    constexpr uint32_t MAX_NSPLIT = 2;
    return (size_t)n_heads * batch * MAX_NSPLIT * head_dim * sizeof(float)
         + (size_t)n_heads * batch * MAX_NSPLIT * 2        * sizeof(float);
}

bool launch_fattn_prefill_mma_gqa_v4_f16(
        float *       out,
        void  *       scratch,
        size_t        scratch_bytes,
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
    if (!(head_dim == 128 || head_dim == 256)) return false;
    if (batch == 0) return true;
    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return false;
    const uint32_t q_per_kv = n_heads / n_kv_heads;
    if (q_per_kv != 6) return false;

    constexpr uint32_t BR = 8;

    static const bool use_cp_async_k = []() {
        const char *e = std::getenv("QW3_PREFILL_FA2_KCPASYNC");
        if (!e) return true;
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0);
    }();
    static const bool use_cp_async_v = []() {
        const char *e = std::getenv("QW3_PREFILL_FA2_VCPASYNC");
        if (!e) return true;
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0);
    }();
    static const bool use_fp16_o = []() {
        const char *e = std::getenv("QW3_PREFILL_FA2_FP16_O");
        if (!e) return false;
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0);
    }();

    const uint32_t n_blocks_q = (batch + BR - 1) / BR;
    // NSPLIT heuristic: same shape as v2's. Kept conservative for first
    // landing — default NSPLIT=1 since the v4 grid is already smaller
    // (n_kv_heads × n_blocks_q vs v2's n_kv_heads*3 × n_blocks_q*2).
    static const uint32_t v4_nsplit_choice = []() -> uint32_t {
        const char *e = std::getenv("QW3_PREFILL_FA2_NSPLIT");
        if (!e) return 1;
        if (std::strcmp(e, "1") == 0) return 1;
        if (std::strcmp(e, "2") == 0) return 2;
        return 1;
    }();
    const uint32_t nsplit = v4_nsplit_choice;
    float *partials = nullptr, *ms_buf = nullptr;
    if (nsplit > 1) {
        const size_t need = (size_t)n_heads * batch * nsplit * head_dim * sizeof(float)
                          + (size_t)n_heads * batch * nsplit * 2        * sizeof(float);
        if (!scratch || scratch_bytes < need) return false;
        partials = static_cast<float *>(scratch);
        ms_buf   = partials + (size_t)n_heads * batch * nsplit * head_dim;
    }
    const dim3 grid(n_kv_heads, n_blocks_q, nsplit);
    const dim3 block(256);
    const size_t smem = fattn_mma_gqa_v4_smem_bytes(head_dim);

    auto launch = [&](auto HD_v, auto NS_v) {
        constexpr uint32_t HD = decltype(HD_v)::value;
        constexpr uint32_t NS = decltype(NS_v)::value;
        auto dispatch = [&](auto K_v, auto V_v, auto FP16O_v) {
            constexpr bool K = decltype(K_v)::value;
            constexpr bool V = decltype(V_v)::value;
            constexpr bool FP16O = decltype(FP16O_v)::value;
            cudaFuncSetAttribute(
                fattn_prefill_mma_gqa_kernel_v4_t<HD, K, V, NS, FP16O>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            fattn_prefill_mma_gqa_kernel_v4_t<HD, K, V, NS, FP16O>
                <<<grid, block, smem, stream>>>(
                    out, partials, ms_buf, q, q_stride,
                    static_cast<const __half *>(k_cache),
                    static_cast<const __half *>(v_cache),
                    n_heads, n_kv_heads, base_seq_len, batch,
                    q_batch_stride, out_batch_stride, scale);
        };
        auto dispatch_fp16o = [&](auto K_v, auto V_v) {
            if (use_fp16_o) dispatch(K_v, V_v, std::true_type{});
            else            dispatch(K_v, V_v, std::false_type{});
        };
        if (use_cp_async_k && use_cp_async_v) {
            dispatch_fp16o(std::true_type{},  std::true_type{});
        } else if (use_cp_async_k) {
            dispatch_fp16o(std::true_type{},  std::false_type{});
        } else if (use_cp_async_v) {
            dispatch_fp16o(std::false_type{}, std::true_type{});
        } else {
            dispatch_fp16o(std::false_type{}, std::false_type{});
        }
        if (NS > 1) {
            const dim3 cgrid(n_heads, batch, 1);
            const dim3 cblock(HD);
            fattn_prefill_mma_gqa_combine_kernel<HD, NS>
                <<<cgrid, cblock, 0, stream>>>(
                    out, partials, ms_buf, batch, out_batch_stride);
        }
    };

    auto launch_with_ns = [&](auto HD_v) {
        if (nsplit == 2) launch(HD_v, std::integral_constant<uint32_t, 2>{});
        else             launch(HD_v, std::integral_constant<uint32_t, 1>{});
    };
    if (head_dim == 256) launch_with_ns(std::integral_constant<uint32_t, 256>{});
    else                 launch_with_ns(std::integral_constant<uint32_t, 128>{});
    return true;
}

size_t fattn_prefill_mma_gqa_v5_scratch_bytes(uint32_t batch, uint32_t n_heads,
                                              uint32_t n_kv_heads,
                                              uint32_t head_dim) {
    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return 0;
    const uint32_t q_per_kv = n_heads / n_kv_heads;
    if (q_per_kv != 6) return 0;
    if (!(head_dim == 128 || head_dim == 256)) return 0;
    constexpr uint32_t MAX_NSPLIT = 2;
    return (size_t)n_heads * batch * MAX_NSPLIT * head_dim * sizeof(float)
         + (size_t)n_heads * batch * MAX_NSPLIT * 2        * sizeof(float);
}

bool launch_fattn_prefill_mma_gqa_v5_f16(
        float *       out,
        void  *       scratch,
        size_t        scratch_bytes,
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
    if (!(head_dim == 128 || head_dim == 256)) return false;
    if (batch == 0) return true;
    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return false;
    const uint32_t q_per_kv = n_heads / n_kv_heads;
    if (q_per_kv != 6) return false;

    constexpr uint32_t NCOLS2  = 2;
    constexpr uint32_t BC      = 32;
    constexpr uint32_t KV_PAD  = 8;
    constexpr uint32_t S_PAD   = 4;
    constexpr uint32_t O_PAD   = 8;

    static const bool use_cp_async_k = []() {
        const char *e = std::getenv("QW3_PREFILL_FA2_KCPASYNC");
        if (!e) return true;
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0);
    }();
    static const bool use_cp_async_v = []() {
        const char *e = std::getenv("QW3_PREFILL_FA2_VCPASYNC");
        if (!e) return true;
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0);
    }();
    // BR knob: default 32 (the lever). BR=16 is supported as an A/B knob to
    // isolate the O→shmem cost from the q-rows-per-CTA win.
    static const uint32_t br_choice = []() -> uint32_t {
        const char *e = std::getenv("QW3_PREFILL_FA2_V5_BR");
        if (!e) return 32;
        if (std::strcmp(e, "16") == 0) return 16;
        if (std::strcmp(e, "32") == 0) return 32;
        return 32;
    }();

    const uint32_t n_blocks_q = (batch + br_choice - 1) / br_choice;

    static const uint32_t v5_nsplit_choice = []() -> uint32_t {
        const char *e = std::getenv("QW3_PREFILL_FA2_NSPLIT");
        if (!e) return 1;
        if (std::strcmp(e, "1") == 0) return 1;
        if (std::strcmp(e, "2") == 0) return 2;
        return 1;
    }();
    const uint32_t nsplit = v5_nsplit_choice;
    float *partials = nullptr, *ms_buf = nullptr;
    if (nsplit > 1) {
        const size_t need = (size_t)n_heads * batch * nsplit * head_dim * sizeof(float)
                          + (size_t)n_heads * batch * nsplit * 2        * sizeof(float);
        if (!scratch || scratch_bytes < need) return false;
        partials = static_cast<float *>(scratch);
        ms_buf   = partials + (size_t)n_heads * batch * nsplit * head_dim;
    }
    const dim3 grid(n_kv_heads * (q_per_kv / NCOLS2), n_blocks_q, nsplit);
    const dim3 block(128);
    const size_t smem = fattn_mma_gqa_v5_smem_bytes(head_dim, BC, br_choice);

    auto launch = [&](auto HD_v, auto BR_v, auto NS_v) {
        constexpr uint32_t HD = decltype(HD_v)::value;
        constexpr uint32_t BR = decltype(BR_v)::value;
        constexpr uint32_t NS = decltype(NS_v)::value;
        auto dispatch = [&](auto K_v, auto V_v) {
            constexpr bool K = decltype(K_v)::value;
            constexpr bool V = decltype(V_v)::value;
            cudaFuncSetAttribute(
                fattn_prefill_mma_gqa_kernel_v5_t<HD, 6, BR, BC, NCOLS2, KV_PAD, S_PAD, O_PAD, K, V, NS>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            fattn_prefill_mma_gqa_kernel_v5_t<HD, 6, BR, BC, NCOLS2, KV_PAD, S_PAD, O_PAD, K, V, NS>
                <<<grid, block, smem, stream>>>(
                    out, partials, ms_buf, q, q_stride,
                    static_cast<const __half *>(k_cache),
                    static_cast<const __half *>(v_cache),
                    n_heads, n_kv_heads, base_seq_len, batch,
                    q_batch_stride, out_batch_stride, scale);
        };
        if (use_cp_async_k && use_cp_async_v) {
            dispatch(std::true_type{},  std::true_type{});
        } else if (use_cp_async_k) {
            dispatch(std::true_type{},  std::false_type{});
        } else if (use_cp_async_v) {
            dispatch(std::false_type{}, std::true_type{});
        } else {
            dispatch(std::false_type{}, std::false_type{});
        }
        if (NS > 1) {
            const dim3 cgrid(n_heads, batch, 1);
            const dim3 cblock(HD);
            fattn_prefill_mma_gqa_combine_kernel<HD, NS>
                <<<cgrid, cblock, 0, stream>>>(
                    out, partials, ms_buf, batch, out_batch_stride);
        }
    };

    auto launch_with_ns_br = [&](auto HD_v, auto BR_v) {
        if (nsplit == 2) launch(HD_v, BR_v, std::integral_constant<uint32_t, 2>{});
        else             launch(HD_v, BR_v, std::integral_constant<uint32_t, 1>{});
    };
    auto launch_with_br = [&](auto HD_v) {
        if (br_choice == 16) launch_with_ns_br(HD_v, std::integral_constant<uint32_t, 16>{});
        else                 launch_with_ns_br(HD_v, std::integral_constant<uint32_t, 32>{});
    };
    if (head_dim == 256) launch_with_br(std::integral_constant<uint32_t, 256>{});
    else                 launch_with_br(std::integral_constant<uint32_t, 128>{});
    return true;
}

} // namespace ported
} // namespace qw3
