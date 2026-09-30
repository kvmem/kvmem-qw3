#include "dflash_cuda.hpp"

#include <cuda_bf16.h>
#include <math_constants.h>
#include <cfloat>
#include <cmath>
#include <climits>

namespace qw3::dflash_cuda {
namespace {

template<class T>
__global__ void capture_kernel(float *out, const T *hidden, uint32_t rows,
                               uint32_t width, uint32_t stride, uint32_t taps, uint32_t tap) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= uint64_t(rows) * width) return;
    const uint64_t row = i / width, col = i % width;
    out[(row * taps + tap) * width + col] = float(hidden[row * stride + col]);
}

__device__ float warp_sum(float x) {
    for (int offset = 16; offset; offset /= 2)
        x += __shfl_down_sync(0xffffffff, x, offset);
    return x;
}

__global__ void conv_kernel(float *out, const float *x, const float *dynamic,
                            const float *base, uint32_t rows, uint32_t hidden,
                            uint32_t block, uint32_t taps, uint32_t group, uint32_t side) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= uint64_t(rows) * hidden) return;
    const uint32_t row = index / hidden, col = index % hidden;
    const uint32_t groups = hidden / group;
    float sum = 0;
    for (uint32_t tap = 0; tap < taps && tap <= row % block; ++tap) {
        const float value = x[index - uint64_t(tap) * hidden];
        const float delta = dynamic[((uint64_t(row) * 2 + side) * taps + tap) * groups + col / group];
        sum += base[(uint64_t(side) * taps + tap) * hidden + col] * value;
        sum = fmaf(delta, value, sum);
    }
    out[index] = sum;
}

// Each CTA handles a query/head. Its warps independently score KV rows, then
// columns reduce the normalized weighted values. KV storage is bounded by the
// drafter window, and no mask/score matrix is allocated in global memory.
template<bool Batched = false>
__global__ void attention_kernel(float *out, const float *q, const float *k,
                                 const float *v, uint32_t queries, uint32_t keys,
                                 uint32_t heads, uint32_t kv_heads,
                                 uint32_t dim, uint32_t window,
                                 const AttentionRequest *requests) {
    if constexpr (Batched) {
        const auto request = requests[blockIdx.z];
        k = request.keys; v = request.values; keys = request.count;
        const uint64_t offset = uint64_t(blockIdx.z) * queries * heads * dim;
        q += offset; out += offset;
    }
    extern __shared__ float score[];
    __shared__ float reduction[256];
    const uint32_t qi = blockIdx.x, head = blockIdx.y;
    const uint32_t kvh = head / (heads / kv_heads);
    const int position = int(keys - queries + qi);
    const int begin = max(0, position - int(window) + 1);
    const int end = min(int(keys), position + int(window));
    const int count = end - begin;
    const uint32_t lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const float *query = q + (uint64_t(qi) * heads + head) * dim;
    for (int local = warp; local < count; local += 8) {
        const float *key = k + (uint64_t(begin + local) * kv_heads + kvh) * dim;
        float dot = 0;
        for (uint32_t d = lane; d < dim; d += 32) dot = fmaf(query[d], key[d], dot);
        dot = warp_sum(dot);
        if (lane == 0) score[local] = dot * rsqrtf(float(dim));
    }
    __syncthreads();
    float maximum = -CUDART_INF_F;
    for (int i = threadIdx.x; i < count; i += blockDim.x) maximum = fmaxf(maximum, score[i]);
    reduction[threadIdx.x] = maximum;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        __syncthreads();
    }
    maximum = reduction[0];
    __syncthreads();
    float sum = 0;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        score[i] = expf(score[i] - maximum);
        sum += score[i];
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float denominator = reduction[0];
    for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
        float result = 0;
        for (int i = 0; i < count; ++i)
            result = fmaf(score[i], v[(uint64_t(begin + i) * kv_heads + kvh) * dim + d], result);
        out[(uint64_t(qi) * heads + head) * dim + d] = result / denominator;
    }
}

__device__ bool better(float a, int ai, float b, int bi) {
    return a > b || (a == b && ai < bi);
}

// One partition per CTA; keep its input in registers while selecting winners.
// The second pass merges only the partition winners, not the full vocabulary.
template<bool Merge>
__global__ void topk_kernel(int32_t *out_ids, float *out_values, const float *input,
                            const int32_t *input_ids, uint32_t width,
                            uint32_t top_k, uint32_t partitions) {
    __shared__ float best[256];
    __shared__ int index[256];
    float values[16];
    int ids[16];
    const uint32_t row = blockIdx.x, partition = blockIdx.y;
    const uint32_t start = Merge ? 0 : partition * 4096;
    for (uint32_t i = 0; i < 16; ++i) {
        const uint32_t col = start + threadIdx.x + i * 256;
        values[i] = col < width ? input[uint64_t(row) * width + col] : -CUDART_INF_F;
        ids[i] = col < width ? (Merge ? input_ids[uint64_t(row) * width + col] : int(col)) : INT_MAX;
    }
    for (uint32_t step = 0; step < top_k; ++step) {
        float value = -CUDART_INF_F;
        int id = INT_MAX;
        for (int i = 0; i < 16; ++i)
            if (better(values[i], ids[i], value, id)) { value = values[i]; id = ids[i]; }
        best[threadIdx.x] = value;
        index[threadIdx.x] = id;
        __syncthreads();
        for (int stride = 128; stride; stride /= 2) {
            if (threadIdx.x < stride && better(best[threadIdx.x + stride], index[threadIdx.x + stride],
                                                best[threadIdx.x], index[threadIdx.x])) {
                best[threadIdx.x] = best[threadIdx.x + stride];
                index[threadIdx.x] = index[threadIdx.x + stride];
            }
            __syncthreads();
        }
        const int winner = index[0];
        if (threadIdx.x == 0) {
            const uint64_t offset = (uint64_t(row) * partitions + partition) * top_k + step;
            out_ids[offset] = winner;
            out_values[offset] = best[0];
        }
        __syncthreads();
        for (int i = 0; i < 16; ++i)
            if (ids[i] == winner) { values[i] = -CUDART_INF_F; ids[i] = INT_MAX; }
    }
}

template<class T> __device__ float as_float(T v) { return float(v); }
template<> __device__ float as_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template<class T>
__global__ void select_kernel(int32_t *tokens, float *probs, const int32_t *candidates,
                              const float *unary, const float *hidden,
                              const T *previous, const T *next, const int32_t *anchors,
                              const float *uniforms, uint32_t positions, uint32_t top_k,
                              uint32_t rank, float temperature) {
    __shared__ float scores[32];
    __shared__ int predecessor;
    const uint32_t lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (threadIdx.x == 0) predecessor = anchors[blockIdx.x];
    __syncthreads();
    for (uint32_t pos = 0; pos < positions; ++pos) {
        const uint64_t row = uint64_t(blockIdx.x) * positions + pos;
        for (uint32_t candidate = warp; candidate < top_k; candidate += 8) {
            const int id = candidates[row * top_k + candidate];
            float pair = 0;
            for (uint32_t d = lane; d < rank; d += 32)
                pair = fmaf(as_float(previous[uint64_t(predecessor) * rank + d]) * hidden[row * rank + d],
                            as_float(next[uint64_t(id) * rank + d]), pair);
            pair = warp_sum(pair);
            if (lane == 0) scores[candidate] = unary[row * top_k + candidate] + pair;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t selected = 0;
            for (uint32_t i = 1; i < top_k; ++i)
                if (scores[i] > scores[selected]) selected = i;
            if (temperature > 0) {
                const float maximum = scores[selected];
                float total = 0;
                for (uint32_t i = 0; i < top_k; ++i) { scores[i] = expf((scores[i] - maximum) / temperature); total += scores[i]; }
                const float draw = uniforms[row];
                float cdf = 0;
                selected = top_k - 1;
                bool found = false;
                for (uint32_t i = 0; i < top_k; ++i) {
                    const float probability = scores[i] / total;
                    probs[row * top_k + i] = probability;
                    cdf += probability;
                    if (!found && draw < cdf) { selected = i; found = true; }
                }
            } else {
                for (uint32_t i = 0; i < top_k; ++i) probs[row * top_k + i] = float(i == selected);
            }
            predecessor = candidates[row * top_k + selected];
            tokens[row] = predecessor;
        }
        __syncthreads();
    }
}
} // namespace

cudaError_t capture_features(float *out, const void *hidden, bool bf16,
                             uint32_t rows, uint32_t width, uint32_t stride,
                             uint32_t taps, uint32_t tap, cudaStream_t stream) {
    const uint32_t grid = (uint64_t(rows) * width + 255) / 256;
    if (bf16) capture_kernel<<<grid, 256, 0, stream>>>(out,
        static_cast<const __nv_bfloat16 *>(hidden), rows, width, stride, taps, tap);
    else capture_kernel<<<grid, 256, 0, stream>>>(out,
        static_cast<const float *>(hidden), rows, width, stride, taps, tap);
    return cudaGetLastError();
}

cudaError_t grouped_conv(float *out, const float *x, const float *dynamic,
                        const float *base, uint32_t rows, uint32_t hidden,
                        uint32_t block, uint32_t taps, uint32_t group,
                        uint32_t side, cudaStream_t stream) {
    conv_kernel<<<(uint64_t(rows) * hidden + 255) / 256, 256, 0, stream>>>(
        out, x, dynamic, base, rows, hidden, block, taps, group, side);
    return cudaGetLastError();
}

cudaError_t attention(float *out, const float *q, const float *k, const float *v,
                      uint32_t queries, uint32_t keys, uint32_t heads,
                      uint32_t kv_heads, uint32_t dim, uint32_t window, cudaStream_t stream) {
    attention_kernel<false><<<dim3(queries, heads), 256, keys * sizeof(float), stream>>>(
        out, q, k, v, queries, keys, heads, kv_heads, dim, window, nullptr);
    return cudaGetLastError();
}

cudaError_t attention_batch(float *out, const float *q, const AttentionRequest *requests,
                            uint32_t blocks, uint32_t queries, uint32_t max_keys,
                            uint32_t heads, uint32_t kv_heads, uint32_t dim,
                            uint32_t window, cudaStream_t stream) {
    attention_kernel<true><<<dim3(queries, heads, blocks), 256, max_keys * sizeof(float), stream>>>(
        out, q, nullptr, nullptr, queries, 0, heads, kv_heads, dim, window, requests);
    return cudaGetLastError();
}

cudaError_t topk(int32_t *ids, float *values, const float *logits,
                 int32_t *scratch_ids, float *scratch_values,
                 uint32_t rows, uint32_t vocab, uint32_t top_k, cudaStream_t stream) {
    const uint32_t parts = (vocab + 4095) / 4096;
    if (parts * top_k > 4096) return cudaErrorInvalidValue;
    topk_kernel<false><<<dim3(rows, parts), 256, 0, stream>>>(scratch_ids, scratch_values, logits, nullptr, vocab, top_k, parts);
    if (auto error = cudaGetLastError(); error != cudaSuccess) return error;
    topk_kernel<true><<<rows, 256, 0, stream>>>(ids, values, scratch_values, scratch_ids, parts * top_k, top_k, 1);
    return cudaGetLastError();
}

cudaError_t select(int32_t *tokens, float *probs, const int32_t *candidates,
                   const float *unary, const float *hidden,
                   const void *predecessor, const void *successor, bool bf16,
                   const int32_t *anchors, const float *uniforms,
                   uint32_t blocks, uint32_t positions, uint32_t top_k,
                   uint32_t rank, float temperature, cudaStream_t stream) {
    if (bf16) select_kernel<<<blocks, 256, 0, stream>>>(tokens, probs, candidates, unary, hidden,
        static_cast<const __nv_bfloat16 *>(predecessor), static_cast<const __nv_bfloat16 *>(successor),
        anchors, uniforms, positions, top_k, rank, temperature);
    else select_kernel<<<blocks, 256, 0, stream>>>(tokens, probs, candidates, unary, hidden,
        static_cast<const float *>(predecessor), static_cast<const float *>(successor),
        anchors, uniforms, positions, top_k, rank, temperature);
    return cudaGetLastError();
}
} // namespace qw3::dflash_cuda
