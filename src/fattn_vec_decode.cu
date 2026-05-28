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
          uint32_t NCOLS2,
          bool USE_CP_ASYNC_K, bool USE_CP_ASYNC_V>
__global__ void
__launch_bounds__(128, 1)
fattn_prefill_mma_gqa_kernel_v2_t(
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

    static_assert(BR == 8,                 "v2 requires BR=8");
    static_assert(BC == 64,                "v2 requires BC=64");
    static_assert(NCOLS2 == 2,             "v2 supports NCOLS2=2");
    static_assert(BR * NCOLS2 == MMA_M,    "BR*NCOLS2 == MMA_M");
    static_assert(Q_PER_KV % NCOLS2 == 0,  "Q_PER_KV must divide by NCOLS2");
    static_assert(HEAD_DIM == 128 || HEAD_DIM == 256, "v2 supports HD 128/256");
    static_assert(HEAD_DIM % MMA_K == 0,   "HEAD_DIM multiple of MMA_K");
    static_assert(BC % MMA_N == 0,         "BC multiple of MMA_N");
    static_assert(BC % NWARPS == 0,        "BC divisible by NWARPS");

    constexpr uint32_t QK_KSTEPS  = HEAD_DIM / MMA_K;          // 16
    constexpr uint32_t QK_N_TILES = (BC / NWARPS) / MMA_N;     // 2
    constexpr uint32_t PV_KSTEPS  = BC / MMA_K;                // 4
    constexpr uint32_t HD_PER_WARP = HEAD_DIM / NWARPS;        // 64
    constexpr uint32_t PV_N_TILES  = HD_PER_WARP / MMA_N;      // 8
    constexpr uint32_t Q_GROUPS_PER_KV = Q_PER_KV / NCOLS2;    // 3

    const uint32_t kvg_idx  = blockIdx.x;
    const uint32_t kv_head  = kvg_idx / Q_GROUPS_PER_KV;
    const uint32_t q_group  = kvg_idx % Q_GROUPS_PER_KV;
    const uint32_t block_q  = blockIdx.y;
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

    // Shmem layout.
    extern __shared__ char smem_raw[];
    char *p = smem_raw;
    __half *s_K     = reinterpret_cast<__half *>(p); p += BC       * HEAD_DIM * sizeof(__half);
    __half *s_V_buf = reinterpret_cast<__half *>(p); p += HEAD_DIM * BC       * sizeof(__half);
    float  *s_S     = reinterpret_cast<float  *>(p); p += MMA_M    * BC       * sizeof(float);
    __half *s_P     = reinterpret_cast<__half *>(p); p += MMA_M    * BC       * sizeof(__half);
    float  *s_m     = reinterpret_cast<float  *>(p); p += MMA_M               * sizeof(float);
    float  *s_l     = reinterpret_cast<float  *>(p); p += MMA_M               * sizeof(float);
    float  *s_alpha = reinterpret_cast<float  *>(p); p += MMA_M               * sizeof(float);

    // ---- Q in registers --------------------------------------------------
    // Q_reg[ks][slot] holds the m=16 a-fragment for ks-step k0=ks*MMA_K.
    // ldmatrix.x4 a-frag distribution: lane t holds rows {t/4, t/4+8} at
    // cols {t%4*2..+1, t%4*2+8..+9}. Mapping to packing m=c*BR+r:
    //   slot 0 = pack2h(s[r_a, k0+in_group*2..+1])    head=head_base+C_A
    //   slot 1 = pack2h(s[r_b, k0+in_group*2..+1])    head=head_base+C_B
    //   slot 2 = pack2h(s[r_a, k0+in_group*2+8..+9])  head=head_base+C_A
    //   slot 3 = pack2h(s[r_b, k0+in_group*2+8..+9])  head=head_base+C_B
    unsigned Q_reg[QK_KSTEPS][4];
    {
        const uint32_t q_idx_a = block_q * BR + r_a;
        const uint32_t q_idx_b = block_q * BR + r_b;
        const bool a_active = (q_idx_a < batch);
        const bool b_active = (q_idx_b < batch);
        const uint32_t head_a = head_base + C_A;
        const uint32_t head_b = head_base + C_B;
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

    if (tid < MMA_M) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }
    float O_acc[PV_N_TILES][4];
    #pragma unroll
    for (uint32_t n = 0; n < PV_N_TILES; ++n) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) O_acc[n][i] = 0.0f;
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
                __half *dst = &s_V_buf[shm_t * HEAD_DIM + base_d];
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

        // QK^T phase. One MMA per (ks, nq) covers both packed c-heads.
        float c_h[QK_N_TILES][4];
        #pragma unroll
        for (uint32_t nq = 0; nq < QK_N_TILES; ++nq) {
            c_h[nq][0]=0.f; c_h[nq][1]=0.f; c_h[nq][2]=0.f; c_h[nq][3]=0.f;
        }
        #pragma unroll
        for (uint32_t ks = 0; ks < QK_KSTEPS; ++ks) {
            const uint32_t k0 = ks * MMA_K;
            #pragma unroll
            for (uint32_t nq = 0; nq < QK_N_TILES; ++nq) {
                const uint32_t kv_col = warp * (BC / NWARPS) + nq * MMA_N + group_id;
                const __half *krow0 = &s_K[kv_col * HEAD_DIM + k0 + in_group * 2];
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

        // Write S to shmem with causal mask. c_h[nq] slots:
        //   .0/.1 → m=group_id     (c=C_A, r=r_a=group_id)
        //   .2/.3 → m=group_id+8   (c=C_B, r=r_b=group_id)
        {
            const uint32_t q_idx_a = block_q * BR + r_a;
            const uint32_t q_idx_b = block_q * BR + r_b;
            const bool a_active = (q_idx_a < batch);
            const bool b_active = (q_idx_b < batch);
            const uint32_t my_max_kv_a = base_seq_len + q_idx_a + 1;
            const uint32_t my_max_kv_b = base_seq_len + q_idx_b + 1;
            const uint32_t m_a = C_A * BR + r_a;     // group_id
            const uint32_t m_b = C_B * BR + r_b;     // group_id + 8
            #pragma unroll
            for (uint32_t nq = 0; nq < QK_N_TILES; ++nq) {
                const uint32_t base_col = warp * (BC / NWARPS) + nq * MMA_N + in_group * 2;
                const uint32_t t_a0 = t0 + base_col;
                const uint32_t t_a1 = t0 + base_col + 1;
                float v00 = (a_active && t_a0 < my_max_kv_a) ? c_h[nq][0] * scale : -INFINITY;
                float v01 = (a_active && t_a1 < my_max_kv_a) ? c_h[nq][1] * scale : -INFINITY;
                float v02 = (b_active && t_a0 < my_max_kv_b) ? c_h[nq][2] * scale : -INFINITY;
                float v03 = (b_active && t_a1 < my_max_kv_b) ? c_h[nq][3] * scale : -INFINITY;
                s_S[m_a * BC + base_col + 0] = v00;
                s_S[m_a * BC + base_col + 1] = v01;
                s_S[m_b * BC + base_col + 0] = v02;
                s_S[m_b * BC + base_col + 1] = v03;
            }
        }

        __syncthreads();

        // Online softmax over m=16 rows × BC=64. 4 warps × 4 rows/warp = 16.
        // Each lane covers BC/WARP_SIZE=2 cols.
        constexpr uint32_t ROWS_PER_WARP = MMA_M / NWARPS;       // 4
        constexpr uint32_t COLS_PER_LANE = BC / WARP_SIZE;       // 2
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

        // Rescale O_acc by per-row alpha. Lane (group_id) → m=group_id (a)
        // and m=group_id+8 (b).
        const float alpha_a = s_alpha[group_id];
        const float alpha_b = s_alpha[group_id + 8];
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            O_acc[n][0] *= alpha_a;
            O_acc[n][1] *= alpha_a;
            O_acc[n][2] *= alpha_b;
            O_acc[n][3] *= alpha_b;
        }

        // Phase B: PV. ldmatrix.x4 on s_P loads m=16 a-frag (both c-heads packed).
        // Single MMA per (n, ks) covers both c-heads.
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            const uint32_t n0 = warp * HD_PER_WARP + n * MMA_N;
            #pragma unroll
            for (uint32_t ks = 0; ks < PV_KSTEPS; ++ks) {
                const uint32_t k0 = ks * MMA_K;
                unsigned b0, b1;
                if (USE_CP_ASYNC_V) {
                    const __half *vbase = &s_V_buf[k0 * HEAD_DIM + n0];
                    mma_detail::ldmatrix_x2_b_trans(b0, b1, vbase, HEAD_DIM);
                } else {
                    const uint32_t out_col = n0 + group_id;
                    const __half *vrow0 = &s_V_buf[out_col * BC + k0 + in_group * 2];
                    b0 = mma_detail::pack2h(vrow0);
                    b1 = mma_detail::pack2h(vrow0 + 8);
                }
                unsigned a0, a1, a2, a3;
                mma_detail::ldmatrix_x4_a(a0, a1, a2, a3,
                                          &s_P[0 * BC + k0], BC);
                mma_detail::mma_m16n8k16_f16f16f32(
                    O_acc[n][0], O_acc[n][1], O_acc[n][2], O_acc[n][3],
                    a0, a1, a2, a3, b0, b1);
            }
        }

        if (USE_CP_ASYNC_V) {
            const uint32_t t0_next = t0 + BC;
            if (t0_next < kv_hi) { issue_V(t0_next); mma_detail::cp_async_commit(); }
        }
    }

    // ---- Final write: divide by l_h and store [batch, n_heads, HEAD_DIM] -
    {
        const uint32_t head_a = head_base + C_A;
        const uint32_t head_b = head_base + C_B;
        const uint32_t q_idx_a = block_q * BR + r_a;
        const uint32_t q_idx_b = block_q * BR + r_b;
        const float l_a = s_l[C_A * BR + r_a];
        const float l_b = s_l[C_B * BR + r_b];
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
                out_a[col + 0] = O_acc[n][0] * inv_la;
                out_a[col + 1] = O_acc[n][1] * inv_la;
            }
            if (out_b) {
                out_b[col + 0] = O_acc[n][2] * inv_lb;
                out_b[col + 1] = O_acc[n][3] * inv_lb;
            }
        }
    }
}

static size_t fattn_mma_gqa_v2_smem_bytes(uint32_t head_dim) {
    constexpr uint32_t BR = 8, BC = 64, NCOLS2 = 2;
    constexpr uint32_t MMA_M = BR * NCOLS2;   // 16
    size_t s = 0;
    s += BC       * head_dim * sizeof(__half);   // K
    s += head_dim * BC       * sizeof(__half);   // V
    s += MMA_M    * BC       * sizeof(float);    // S
    s += MMA_M    * BC       * sizeof(__half);   // P
    s += MMA_M               * sizeof(float) * 3;// m, l, alpha
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

// v2 launcher: BR=8 + ncols2=2 + Q-in-regs. NSPLIT=1 always. Currently
// gated to Qwen 3.6 27B's q_per_kv=6.
bool launch_fattn_prefill_mma_gqa_v2_f16(
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
    constexpr uint32_t BR = 8, BC = 64, NCOLS2 = 2;
    const uint32_t n_blocks_q = (batch + BR - 1) / BR;
    const uint32_t q_groups   = q_per_kv / NCOLS2;     // 3
    const dim3 grid(n_kv_heads * q_groups, n_blocks_q, 1);
    const dim3 block(128);
    const size_t smem = fattn_mma_gqa_v2_smem_bytes(head_dim);

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

    auto launch = [&](auto HD_v) {
        constexpr uint32_t HD = decltype(HD_v)::value;
        auto dispatch = [&](auto K_v, auto V_v) {
            constexpr bool K = decltype(K_v)::value;
            constexpr bool V = decltype(V_v)::value;
            cudaFuncSetAttribute(
                fattn_prefill_mma_gqa_kernel_v2_t<HD, 6, BR, BC, NCOLS2, K, V>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            fattn_prefill_mma_gqa_kernel_v2_t<HD, 6, BR, BC, NCOLS2, K, V>
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
    if (head_dim == 256) launch(std::integral_constant<uint32_t, 256>{});
    else                 launch(std::integral_constant<uint32_t, 128>{});
    return true;
}

} // namespace ported
} // namespace qw3
