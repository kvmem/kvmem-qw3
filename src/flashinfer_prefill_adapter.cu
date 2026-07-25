#include "flashinfer_prefill_adapter.hpp"

#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/pos_enc.cuh>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

namespace qw3 {
namespace flashinfer_adapter {
namespace {

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ half from_float<half>(float value) {
    return __float2half(value);
}

template <typename T>
__device__ __forceinline__ float to_float(T value);

template <>
__device__ __forceinline__ float to_float<half>(half value) {
    return __half2float(value);
}

template <typename T>
__global__ void pack_q_kernel(T *q_packed,
                              const float *q,
                              uint32_t q_stride,
                              uint32_t batch,
                              uint32_t n_heads,
                              uint32_t head_dim,
                              uint32_t q_batch_stride) {
    const uint64_t total = static_cast<uint64_t>(batch) * n_heads * head_dim;
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;

    const uint32_t d = static_cast<uint32_t>(i % head_dim);
    const uint64_t tmp = i / head_dim;
    const uint32_t h = static_cast<uint32_t>(tmp % n_heads);
    const uint32_t b = static_cast<uint32_t>(tmp / n_heads);
    q_packed[i] = from_float<T>(q[static_cast<uint64_t>(b) * q_batch_stride
                                 + static_cast<uint64_t>(h) * q_stride + d]);
}

// Gate-fused unpack: out = sigmoid(gate) * o. qw3_ly path.
template <typename T>
__global__ void unpack_gate_kernel(float *out,
                                   const T *o_packed,
                                   const float *q,
                                   uint32_t q_stride,
                                   uint32_t batch,
                                   uint32_t n_heads,
                                   uint32_t head_dim,
                                   uint32_t q_batch_stride,
                                   uint32_t out_batch_stride) {
    const uint64_t total = static_cast<uint64_t>(batch) * n_heads * head_dim;
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;

    const uint32_t d = static_cast<uint32_t>(i % head_dim);
    const uint64_t tmp = i / head_dim;
    const uint32_t h = static_cast<uint32_t>(tmp % n_heads);
    const uint32_t b = static_cast<uint32_t>(tmp / n_heads);
    const uint64_t q_off = static_cast<uint64_t>(b) * q_batch_stride
                         + static_cast<uint64_t>(h) * q_stride;
    const float value = to_float<T>(o_packed[i]);
    const float gate = q[q_off + head_dim + d];
    out[static_cast<uint64_t>(b) * out_batch_stride
        + static_cast<uint64_t>(h) * head_dim + d] =
        value * (1.0f / (1.0f + expf(-gate)));
}

// Plain unpack: out = fp16 -> fp32. Caller still applies the sigmoid gate.
template <typename T>
__global__ void unpack_kernel(float *out,
                              const T *o_packed,
                              uint32_t batch,
                              uint32_t n_heads,
                              uint32_t head_dim,
                              uint32_t out_batch_stride) {
    const uint64_t total = static_cast<uint64_t>(batch) * n_heads * head_dim;
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;

    const uint32_t d = static_cast<uint32_t>(i % head_dim);
    const uint64_t tmp = i / head_dim;
    const uint32_t h = static_cast<uint32_t>(tmp % n_heads);
    const uint32_t b = static_cast<uint32_t>(tmp / n_heads);
    out[static_cast<uint64_t>(b) * out_batch_stride
        + static_cast<uint64_t>(h) * head_dim + d] = to_float<T>(o_packed[i]);
}

// __FA_PORT_TAIL__
template <typename TQ, typename TKV, typename TO>
bool run_prefill_typed(
        TQ *q_packed,
        TO *o_packed,
        const float *q, uint32_t q_stride,
        const void *k_cache, const void *v_cache,
        uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
        uint32_t batch, uint32_t base_seq_len,
        uint32_t q_batch_stride,
        float scale, cudaStream_t stream,
        unsigned &threads_out, unsigned &blocks_out) {
    if (head_dim != 256) return false;
    if (n_heads == 0 || n_kv_heads == 0 || (n_heads % n_kv_heads) != 0) return false;
    threads_out = 256;
    if (batch == 0) {
        blocks_out = 0;
        return true;
    }
    const uint64_t q_elems = static_cast<uint64_t>(batch) * n_heads * head_dim;
    blocks_out = static_cast<unsigned>((q_elems + threads_out - 1) / threads_out);
    pack_q_kernel<TQ><<<blocks_out, threads_out, 0, stream>>>(
        q_packed, q, q_stride, batch, n_heads, head_dim, q_batch_stride);
    if (cudaGetLastError() != cudaSuccess) return false;

    using Params = flashinfer::SinglePrefillParams<TQ, TKV, TO>;
    using Variant = flashinfer::DefaultAttention<false, false, false, false>;
    Params params(
        q_packed,
        const_cast<TKV *>(static_cast<const TKV *>(k_cache)),
        const_cast<TKV *>(static_cast<const TKV *>(v_cache)),
        nullptr, o_packed, nullptr, nullptr,
        n_heads, n_kv_heads, batch, base_seq_len + batch,
        n_heads * head_dim, head_dim,
        n_kv_heads * head_dim, head_dim, head_dim,
        -1, 0.0f, scale, 1.0f, 1.0f);
    cudaError_t st = flashinfer::SinglePrefillWithKVCacheDispatched<
        256, 256,
        flashinfer::PosEncodingMode::kNone,
        false,
        flashinfer::MaskMode::kCausal,
        Variant>(params, nullptr, stream);
    return st == cudaSuccess;
}

constexpr uint32_t kPagedPrefillPrefixI32 = 16;

bool paged_prefill_plan_cache_enabled() {
    const char *env = std::getenv("QW3_FLASHINFER_PLAN_CACHE");
    return env == nullptr ||
           (std::strcmp(env, "0") != 0 &&
            std::strcmp(env, "false") != 0 &&
            std::strcmp(env, "off") != 0);
}

// Every normal-attention layer in one prefill chunk has the same scheduling
// shape and shares the same persistent FlashInfer workspaces. PrefillPlan is
// independent of the actual K/V page IDs, but the old path rebuilt and
// re-uploaded that identical plan once per layer. Cache only an immediately
// reusable workspace/shape tuple; a changed chunk length, KV-page count,
// workspace, stream, or dtype instantiation is a miss and replans normally.
struct PagedPrefillPlanCache {
    bool valid = false;
    int32_t *int_workspace = nullptr;
    size_t int_workspace_bytes = 0;
    float *requested_float_workspace = nullptr;
    size_t requested_float_workspace_bytes = 0;
    float *active_float_workspace = nullptr;
    size_t active_float_workspace_bytes = 0;
    cudaStream_t stream = nullptr;
    uint32_t batch = 0;
    uint32_t need_pages = 0;
    uint32_t n_heads = 0;
    uint32_t n_kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t page_size = 0;
    bool allow_split_kv = false;
    uint64_t consecutive_hits = 0;
    uint64_t total_hits = 0;
    uint64_t total_misses = 0;
    flashinfer::PrefillPlanInfo plan_info;

    bool matches(
            int32_t *iw, size_t iw_bytes,
            float *fw, size_t fw_bytes, cudaStream_t s,
            uint32_t b, uint32_t pages, uint32_t nh,
            uint32_t nkh, uint32_t hd, uint32_t ps,
            bool allow_split) const {
        return valid &&
               int_workspace == iw &&
               int_workspace_bytes == iw_bytes &&
               requested_float_workspace == fw &&
               requested_float_workspace_bytes == fw_bytes &&
               stream == s &&
               batch == b &&
               need_pages == pages &&
               n_heads == nh &&
               n_kv_heads == nkh &&
               head_dim == hd &&
               page_size == ps &&
               allow_split_kv == allow_split;
    }
};

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM, typename Params, typename Variant>
cudaError_t dispatch_batch_prefill_paged(Params params,
                                         typename Params::DTypeO *tmp_v,
                                         float *tmp_s,
                                         cudaStream_t stream) {
    return flashinfer::BatchPrefillWithPagedKVCacheDispatched<
        CTA_TILE_Q, HEAD_DIM, HEAD_DIM,
        flashinfer::PosEncodingMode::kNone,
        false,
        flashinfer::MaskMode::kCausal,
        Variant>(params, tmp_v, tmp_s, false, stream);
}

template <typename TKV>
bool run_batch_prefill_paged_typed(
        float *out,
        half *q_f16,
        half *o_f16,
        int32_t *int_workspace,
        void *host_int_workspace,
        size_t int_workspace_bytes,
        float *float_workspace,
        size_t float_workspace_bytes,
        const float *q,
        uint32_t q_stride,
        const void *k_cache,
        const void *v_cache,
        const int32_t *page_indices,
        uint32_t n_pages,
        uint32_t page_size,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t base_seq_len,
        uint32_t batch,
        uint32_t q_batch_stride,
        uint32_t out_batch_stride,
        float scale,
        cudaStream_t stream,
        unsigned &threads_out,
        unsigned &blocks_out,
        bool *plan_cache_hit_out) {
    if (plan_cache_hit_out) *plan_cache_hit_out = false;
    if (out == nullptr || q_f16 == nullptr || o_f16 == nullptr ||
        int_workspace == nullptr || host_int_workspace == nullptr ||
        q == nullptr || k_cache == nullptr || v_cache == nullptr ||
        page_indices == nullptr) {
        return false;
    }
    if (head_dim != 128 && head_dim != 256) return false;
    if (n_heads == 0 || n_kv_heads == 0 || (n_heads % n_kv_heads) != 0) return false;
    if (batch == 0 || page_size == 0 || q_stride < head_dim) return false;

    const uint32_t kv_len = base_seq_len + batch;
    const uint32_t need_pages = std::max<uint32_t>((kv_len + page_size - 1U) / page_size, 1U);
    if (need_pages > n_pages) return false;
    const uint32_t last_len = kv_len - (need_pages - 1U) * page_size;

    const size_t prefix_bytes = kPagedPrefillPrefixI32 * sizeof(int32_t);
    if (int_workspace_bytes <= prefix_bytes) return false;

    int32_t *host_i32 = static_cast<int32_t *>(host_int_workspace);
    std::memset(host_i32, 0, prefix_bytes);
    host_i32[0] = 0;
    host_i32[1] = static_cast<int32_t>(batch);
    host_i32[2] = 0;
    host_i32[3] = static_cast<int32_t>(need_pages);
    host_i32[4] = static_cast<int32_t>(last_len);
    if (cudaMemcpyAsync(int_workspace, host_i32, prefix_bytes,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return false;
    }

    threads_out = 256;
    const uint64_t q_elems = static_cast<uint64_t>(batch) * n_heads * head_dim;
    blocks_out = static_cast<unsigned>((q_elems + threads_out - 1) / threads_out);
    pack_q_kernel<half><<<blocks_out, threads_out, 0, stream>>>(
        q_f16, q, q_stride, batch, n_heads, head_dim, q_batch_stride);
    if (cudaGetLastError() != cudaSuccess) return false;

    using Params = flashinfer::BatchPrefillPagedParams<half, TKV, half, int32_t>;
    using Variant = flashinfer::DefaultAttention<false, false, false, false>;

    flashinfer::PrefillPlanInfo plan_info;
    int32_t qo_indptr_h[2] = {0, static_cast<int32_t>(batch)};
    int32_t kv_indptr_h[2] = {0, static_cast<int32_t>(need_pages)};
    void *plan_int_workspace =
        reinterpret_cast<void *>(reinterpret_cast<char *>(int_workspace) + prefix_bytes);
    void *plan_host_workspace =
        reinterpret_cast<void *>(reinterpret_cast<char *>(host_int_workspace) + prefix_bytes);
    const size_t plan_int_workspace_bytes = int_workspace_bytes - prefix_bytes;
    // Enable KV-cache partitioning when a float scratch is supplied. This is
    // the verify (small qo, huge kv) case: without splitting the planner packs
    // all qo rows into 1 tile and launches only n_kv_heads CTAs (~4 on a 188-SM
    // GPU at 128K), serializing the entire KV stream. Splitting fans the KV
    // across hundreds of CTAs; FlashInfer's kernel writes per-chunk partials to
    // tmp_v/tmp_s and merges them via VariableLengthMergeStates internally.
    const bool allow_split_kv =
        (float_workspace != nullptr && float_workspace_bytes > 0);
    float *active_float_workspace = float_workspace;
    size_t active_float_workspace_bytes = float_workspace_bytes;
    static thread_local PagedPrefillPlanCache plan_cache;
    const bool use_plan_cache = paged_prefill_plan_cache_enabled();
    const bool plan_cache_hit =
        use_plan_cache &&
        plan_cache.matches(
            int_workspace, int_workspace_bytes,
            float_workspace, float_workspace_bytes, stream,
            batch, need_pages, n_heads, n_kv_heads, head_dim,
            page_size, allow_split_kv);
    if (plan_cache_hit_out) *plan_cache_hit_out = plan_cache_hit;
    auto plan = [&](bool disable_split_kv) {
        return flashinfer::PrefillPlan<int32_t>(
            disable_split_kv ? nullptr : active_float_workspace,
            disable_split_kv ? 0 : active_float_workspace_bytes,
            plan_int_workspace,
            plan_host_workspace,
            plan_int_workspace_bytes,
            plan_info,
            qo_indptr_h,
            kv_indptr_h,
            batch,
            1,
            n_heads,
            n_kv_heads,
            head_dim,
            head_dim,
            page_size,
            false,
            sizeof(half),
            -1,
            0,
            disable_split_kv,
            0,
            stream);
    };
    if (plan_cache_hit) {
        ++plan_cache.consecutive_hits;
        ++plan_cache.total_hits;
        plan_info = plan_cache.plan_info;
        active_float_workspace = plan_cache.active_float_workspace;
        active_float_workspace_bytes =
            plan_cache.active_float_workspace_bytes;
    } else {
        if (plan_cache.valid &&
            std::getenv("QW3_FLASHINFER_PLAN_CACHE_TRACE")) {
            std::fprintf(
                stderr,
                "[flashinfer-plan-cache] tkv_bytes=%zu hits=%llu "
                "total_hits=%llu total_misses=%llu batch=%u pages=%u\n",
                sizeof(TKV),
                static_cast<unsigned long long>(
                    plan_cache.consecutive_hits),
                static_cast<unsigned long long>(plan_cache.total_hits),
                static_cast<unsigned long long>(
                    plan_cache.total_misses),
                plan_cache.batch, plan_cache.need_pages);
        }
        plan_cache.consecutive_hits = 0;
        ++plan_cache.total_misses;
        cudaError_t plan_st = cudaErrorInvalidValue;
        try {
            plan_st = plan(/*disable_split_kv=*/!allow_split_kv);
        } catch (const std::exception &) {
            // FlashInfer's aligned workspace allocator throws when a requested
            // split plan does not fit. Correctness does not require splitting:
            // retry with the non-split planner instead of failing the request.
            active_float_workspace = nullptr;
            active_float_workspace_bytes = 0;
            plan_info = {};
            try {
                plan_st = plan(/*disable_split_kv=*/true);
            } catch (const std::exception &) {
                return false;
            }
        }
        if (plan_st != cudaSuccess) return false;
        if (use_plan_cache) {
            plan_cache.valid = true;
            plan_cache.int_workspace = int_workspace;
            plan_cache.int_workspace_bytes = int_workspace_bytes;
            plan_cache.requested_float_workspace = float_workspace;
            plan_cache.requested_float_workspace_bytes =
                float_workspace_bytes;
            plan_cache.active_float_workspace = active_float_workspace;
            plan_cache.active_float_workspace_bytes =
                active_float_workspace_bytes;
            plan_cache.stream = stream;
            plan_cache.batch = batch;
            plan_cache.need_pages = need_pages;
            plan_cache.n_heads = n_heads;
            plan_cache.n_kv_heads = n_kv_heads;
            plan_cache.head_dim = head_dim;
            plan_cache.page_size = page_size;
            plan_cache.allow_split_kv = allow_split_kv;
            plan_cache.plan_info = plan_info;
        }
    }

    int32_t *q_indptr_d = int_workspace;
    int32_t *page_indptr_d = int_workspace + 2;
    int32_t *last_page_len_d = int_workspace + 4;
    int32_t *plan_i32 = reinterpret_cast<int32_t *>(plan_int_workspace);

    auto paged_kv = flashinfer::paged_kv_t<TKV, int32_t>(
        n_kv_heads, page_size, head_dim, 1, flashinfer::QKVLayout::kNHD,
        const_cast<TKV *>(static_cast<const TKV *>(k_cache)),
        const_cast<TKV *>(static_cast<const TKV *>(v_cache)),
        const_cast<int32_t *>(page_indices),
        page_indptr_d,
        last_page_len_d,
        nullptr);
    Params params(q_f16,
                  paged_kv,
                  nullptr,
                  q_indptr_d,
                  nullptr,
                  nullptr,
                  o_f16,
                  nullptr,
                  nullptr,
                  n_heads,
                  static_cast<int32_t>(n_heads * head_dim),
                  static_cast<int32_t>(head_dim),
                  -1,
                  0.0f,
                  scale,
                  1.0f,
                  1.0f);
    params.request_indices = reinterpret_cast<int32_t *>(
        reinterpret_cast<char *>(plan_i32) + plan_info.request_indices_offset);
    params.qo_tile_indices = reinterpret_cast<int32_t *>(
        reinterpret_cast<char *>(plan_i32) + plan_info.qo_tile_indices_offset);
    params.kv_tile_indices = reinterpret_cast<int32_t *>(
        reinterpret_cast<char *>(plan_i32) + plan_info.kv_tile_indices_offset);
    params.o_indptr = reinterpret_cast<int32_t *>(
        reinterpret_cast<char *>(plan_i32) + plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr = reinterpret_cast<int32_t *>(
        reinterpret_cast<char *>(plan_i32) + plan_info.kv_chunk_size_ptr_offset);
    params.padded_batch_size = static_cast<uint32_t>(plan_info.padded_batch_size);
    params.max_total_num_rows = batch;

    // Partial-output scratch for the split-KV path. Non-null tmp_v switches the
    // FlashInfer kernel into partition_kv mode: each (qo_tile, kv_chunk) CTA
    // writes its partial O + LSE into tmp_v/tmp_s, then the kernel merges them
    // with VariableLengthMergeStates using merge_indptr.
    typename Params::DTypeO *tmp_v = nullptr;
    float *tmp_s = nullptr;
    if (plan_info.split_kv) {
        params.partition_kv = true;
        params.merge_indptr = reinterpret_cast<int32_t *>(
            reinterpret_cast<char *>(plan_i32) + plan_info.merge_indptr_offset);
        params.block_valid_mask = reinterpret_cast<bool *>(
            reinterpret_cast<char *>(plan_i32) + plan_info.block_valid_mask_offset);
        params.total_num_rows = nullptr;
        tmp_v = reinterpret_cast<typename Params::DTypeO *>(
            reinterpret_cast<char *>(active_float_workspace) + plan_info.v_offset);
        tmp_s = reinterpret_cast<float *>(
            reinterpret_cast<char *>(active_float_workspace) + plan_info.s_offset);
    } else {
        params.partition_kv = false;
        params.total_num_rows = nullptr;
        params.merge_indptr = nullptr;
        params.block_valid_mask = nullptr;
    }

    cudaError_t st = cudaErrorInvalidValue;
    if (head_dim == 128) {
        if (plan_info.cta_tile_q == 16) {
            st = dispatch_batch_prefill_paged<16, 128, Params, Variant>(
                params, tmp_v, tmp_s, stream);
        } else if (plan_info.cta_tile_q == 64) {
            st = dispatch_batch_prefill_paged<64, 128, Params, Variant>(
                params, tmp_v, tmp_s, stream);
        } else if (plan_info.cta_tile_q == 128) {
            st = dispatch_batch_prefill_paged<128, 128, Params, Variant>(
                params, tmp_v, tmp_s, stream);
        }
    } else if (head_dim == 256) {
        if (plan_info.cta_tile_q == 16) {
            st = dispatch_batch_prefill_paged<16, 256, Params, Variant>(
                params, tmp_v, tmp_s, stream);
        } else if (plan_info.cta_tile_q == 64) {
            st = dispatch_batch_prefill_paged<64, 256, Params, Variant>(
                params, tmp_v, tmp_s, stream);
        }
    }
    return st == cudaSuccess;
}

template <typename TKV>
bool run_batch_prefill_paged_ragged_typed(
        float *out,
        half *q_f16,
        half *o_f16,
        int32_t *int_workspace,
        void *host_int_workspace,
        size_t int_workspace_bytes,
        float *float_workspace,
        size_t float_workspace_bytes,
        const float *q,
        uint32_t q_stride,
        const void *k_cache,
        const void *v_cache,
        const int32_t *page_indices,
        const int32_t *page_indptr,
        const int32_t *last_page_len,
        const int32_t *q_indptr,
        const int32_t *q_indptr_host,
        const int32_t *page_indptr_host,
        uint32_t batch,
        uint32_t total_q,
        uint32_t page_size,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t q_batch_stride,
        uint32_t out_batch_stride,
        float scale,
        cudaStream_t stream,
        unsigned &threads_out,
        unsigned &blocks_out) {
    if (out == nullptr || q_f16 == nullptr || o_f16 == nullptr ||
        int_workspace == nullptr || host_int_workspace == nullptr ||
        q == nullptr || k_cache == nullptr || v_cache == nullptr ||
        page_indices == nullptr || page_indptr == nullptr ||
        last_page_len == nullptr || q_indptr == nullptr ||
        q_indptr_host == nullptr || page_indptr_host == nullptr) {
        return false;
    }
    if (head_dim != 128 && head_dim != 256) return false;
    if (n_heads == 0 || n_kv_heads == 0 || (n_heads % n_kv_heads) != 0) return false;
    if (batch == 0 || total_q == 0 || page_size == 0 || q_stride < head_dim) return false;
    if (q_indptr_host[0] != 0 || page_indptr_host[0] != 0) return false;
    if (q_indptr_host[batch] != static_cast<int32_t>(total_q)) return false;
    if (page_indptr_host[batch] <= 0) return false;

    threads_out = 256;
    const uint64_t q_elems = static_cast<uint64_t>(total_q) * n_heads * head_dim;
    blocks_out = static_cast<unsigned>((q_elems + threads_out - 1) / threads_out);
    pack_q_kernel<half><<<blocks_out, threads_out, 0, stream>>>(
        q_f16, q, q_stride, total_q, n_heads, head_dim, q_batch_stride);
    if (cudaGetLastError() != cudaSuccess) return false;

    using Params = flashinfer::BatchPrefillPagedParams<half, TKV, half, int32_t>;
    using Variant = flashinfer::DefaultAttention<false, false, false, false>;

    flashinfer::PrefillPlanInfo plan_info;
    // For MTP verify the "batch" is only a handful of query rows per request,
    // but each row attends its own long KV. Without split-KV the planner packs
    // all rows into a tiny qo grid and launches ~n_kv_heads CTAs that serially
    // stream each row's KV — so co-batching N concurrent requests does NOT speed
    // up the dominant attention cost. Enable split-KV when a float scratch is
    // supplied; the kernel writes per-chunk partials to tmp_v/tmp_s and merges
    // them via VariableLengthMergeStates (mirrors the non-ragged verify path).
    const bool allow_split_kv =
        (float_workspace != nullptr && float_workspace_bytes > 0);
    float *active_float_workspace = float_workspace;
    size_t active_float_workspace_bytes = float_workspace_bytes;
    auto plan = [&](bool disable_split_kv) {
        return flashinfer::PrefillPlan<int32_t>(
            disable_split_kv ? nullptr : active_float_workspace,
            disable_split_kv ? 0 : active_float_workspace_bytes,
            int_workspace,
            host_int_workspace,
            int_workspace_bytes,
            plan_info,
            const_cast<int32_t *>(q_indptr_host),
            const_cast<int32_t *>(page_indptr_host),
            total_q,
            batch,
            n_heads,
            n_kv_heads,
            head_dim,
            head_dim,
            page_size,
            false,
            sizeof(half),
            -1,
            0,
            disable_split_kv,
            0,
            stream);
    };
    cudaError_t plan_st = cudaErrorInvalidValue;
    try {
        plan_st = plan(/*disable_split_kv=*/!allow_split_kv);
    } catch (const std::exception &) {
        active_float_workspace = nullptr;
        active_float_workspace_bytes = 0;
        plan_info = {};
        try {
            plan_st = plan(/*disable_split_kv=*/true);
        } catch (const std::exception &) {
            return false;
        }
    }
    if (plan_st != cudaSuccess) return false;

    auto paged_kv = flashinfer::paged_kv_t<TKV, int32_t>(
        n_kv_heads, page_size, head_dim, batch, flashinfer::QKVLayout::kNHD,
        const_cast<TKV *>(static_cast<const TKV *>(k_cache)),
        const_cast<TKV *>(static_cast<const TKV *>(v_cache)),
        const_cast<int32_t *>(page_indices),
        const_cast<int32_t *>(page_indptr),
        const_cast<int32_t *>(last_page_len),
        nullptr);
    Params params(q_f16,
                  paged_kv,
                  nullptr,
                  const_cast<int32_t *>(q_indptr),
                  nullptr,
                  nullptr,
                  o_f16,
                  nullptr,
                  nullptr,
                  n_heads,
                  static_cast<int32_t>(n_heads * head_dim),
                  static_cast<int32_t>(head_dim),
                  -1,
                  0.0f,
                  scale,
                  1.0f,
                  1.0f);
    int32_t *plan_i32 = int_workspace;
    params.request_indices = reinterpret_cast<int32_t *>(
        reinterpret_cast<char *>(plan_i32) + plan_info.request_indices_offset);
    params.qo_tile_indices = reinterpret_cast<int32_t *>(
        reinterpret_cast<char *>(plan_i32) + plan_info.qo_tile_indices_offset);
    params.kv_tile_indices = reinterpret_cast<int32_t *>(
        reinterpret_cast<char *>(plan_i32) + plan_info.kv_tile_indices_offset);
    params.o_indptr = reinterpret_cast<int32_t *>(
        reinterpret_cast<char *>(plan_i32) + plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr = reinterpret_cast<int32_t *>(
        reinterpret_cast<char *>(plan_i32) + plan_info.kv_chunk_size_ptr_offset);
    params.padded_batch_size = static_cast<uint32_t>(plan_info.padded_batch_size);
    params.max_total_num_rows = total_q;

    typename Params::DTypeO *tmp_v = nullptr;
    float *tmp_s = nullptr;
    if (plan_info.split_kv) {
        params.partition_kv = true;
        params.merge_indptr = reinterpret_cast<int32_t *>(
            reinterpret_cast<char *>(plan_i32) + plan_info.merge_indptr_offset);
        params.block_valid_mask = reinterpret_cast<bool *>(
            reinterpret_cast<char *>(plan_i32) + plan_info.block_valid_mask_offset);
        params.total_num_rows = nullptr;
        tmp_v = reinterpret_cast<typename Params::DTypeO *>(
            reinterpret_cast<char *>(active_float_workspace) + plan_info.v_offset);
        tmp_s = reinterpret_cast<float *>(
            reinterpret_cast<char *>(active_float_workspace) + plan_info.s_offset);
    } else {
        params.partition_kv = false;
        params.total_num_rows = nullptr;
        params.merge_indptr = nullptr;
        params.block_valid_mask = nullptr;
    }

    cudaError_t st = cudaErrorInvalidValue;
    if (head_dim == 128) {
        if (plan_info.cta_tile_q == 16) {
            st = dispatch_batch_prefill_paged<16, 128, Params, Variant>(
                params, tmp_v, tmp_s, stream);
        } else if (plan_info.cta_tile_q == 64) {
            st = dispatch_batch_prefill_paged<64, 128, Params, Variant>(
                params, tmp_v, tmp_s, stream);
        } else if (plan_info.cta_tile_q == 128) {
            st = dispatch_batch_prefill_paged<128, 128, Params, Variant>(
                params, tmp_v, tmp_s, stream);
        }
    } else if (head_dim == 256) {
        if (plan_info.cta_tile_q == 16) {
            st = dispatch_batch_prefill_paged<16, 256, Params, Variant>(
                params, tmp_v, tmp_s, stream);
        } else if (plan_info.cta_tile_q == 64) {
            st = dispatch_batch_prefill_paged<64, 256, Params, Variant>(
                params, tmp_v, tmp_s, stream);
        }
    }
    return st == cudaSuccess;
}

} // namespace

bool launch_prefill_f16q_f16kv_gated(
        float *out,
        __half *q_f16,
        __half *o_f16,
        const float *q, uint32_t q_stride,
        const void *k_cache, const void *v_cache,
        uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
        uint32_t batch, uint32_t base_seq_len,
        uint32_t q_batch_stride, uint32_t out_batch_stride,
        float scale, cudaStream_t stream) {
    unsigned threads = 0, blocks = 0;
    if (!run_prefill_typed<half, half, half>(
            q_f16, o_f16, q, q_stride, k_cache, v_cache,
            n_heads, n_kv_heads, head_dim, batch, base_seq_len,
            q_batch_stride, scale, stream, threads, blocks)) {
        return false;
    }
    if (batch == 0) return true;
    unpack_gate_kernel<half><<<blocks, threads, 0, stream>>>(
        out, o_f16, q, q_stride, batch, n_heads, head_dim,
        q_batch_stride, out_batch_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool launch_prefill_f16q_f16kv(
        float *out,
        __half *q_f16,
        __half *o_f16,
        const float *q, uint32_t q_stride,
        const void *k_cache, const void *v_cache,
        uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
        uint32_t batch, uint32_t base_seq_len,
        uint32_t q_batch_stride, uint32_t out_batch_stride,
        float scale, cudaStream_t stream) {
    unsigned threads = 0, blocks = 0;
    if (!run_prefill_typed<half, half, half>(
            q_f16, o_f16, q, q_stride, k_cache, v_cache,
            n_heads, n_kv_heads, head_dim, batch, base_seq_len,
            q_batch_stride, scale, stream, threads, blocks)) {
        return false;
    }
    if (batch == 0) return true;
    unpack_kernel<half><<<blocks, threads, 0, stream>>>(
        out, o_f16, batch, n_heads, head_dim, out_batch_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool launch_prefill_f16q_fp8kv(
        float *out,
        __half *q_f16,
        __half *o_f16,
        const float *q, uint32_t q_stride,
        const void *k_cache, const void *v_cache,
        uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
        uint32_t batch, uint32_t base_seq_len,
        uint32_t q_batch_stride, uint32_t out_batch_stride,
        float scale, cudaStream_t stream) {
    unsigned threads = 0, blocks = 0;
    const bool launched = run_prefill_typed<half, __nv_fp8_e4m3, half>(
        q_f16, o_f16, q, q_stride, k_cache, v_cache,
        n_heads, n_kv_heads, head_dim, batch, base_seq_len,
        q_batch_stride, scale, stream, threads, blocks);
    if (!launched) return false;
    if (batch == 0) return true;
    unpack_kernel<half><<<blocks, threads, 0, stream>>>(
        out, o_f16, batch, n_heads, head_dim, out_batch_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool launch_batch_prefill_paged_f16q_f16kv_gated(
        float *out,
        half *q_f16,
        half *o_f16,
        int32_t *int_workspace,
        void *host_int_workspace,
        size_t int_workspace_bytes,
        float *float_workspace,
        size_t float_workspace_bytes,
        const float *q,
        uint32_t q_stride,
        const void *k_cache,
        const void *v_cache,
        const int32_t *page_indices,
        uint32_t n_pages,
        uint32_t page_size,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t base_seq_len,
        uint32_t batch,
        uint32_t q_batch_stride,
        uint32_t out_batch_stride,
        float scale,
        cudaStream_t stream,
        bool *plan_cache_hit_out) {
    unsigned threads = 0, blocks = 0;
    if (!run_batch_prefill_paged_typed<half>(
            out, q_f16, o_f16, int_workspace, host_int_workspace,
            int_workspace_bytes, float_workspace, float_workspace_bytes,
            q, q_stride, k_cache, v_cache, page_indices,
            n_pages, page_size, n_heads, n_kv_heads, head_dim, base_seq_len,
            batch, q_batch_stride, out_batch_stride, scale, stream, threads,
            blocks, plan_cache_hit_out)) {
        return false;
    }
    unpack_gate_kernel<half><<<blocks, threads, 0, stream>>>(
        out, o_f16, q, q_stride, batch, n_heads, head_dim,
        q_batch_stride, out_batch_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool launch_batch_prefill_paged_f16q_fp8kv_gated(
        float *out,
        half *q_f16,
        half *o_f16,
        int32_t *int_workspace,
        void *host_int_workspace,
        size_t int_workspace_bytes,
        float *float_workspace,
        size_t float_workspace_bytes,
        const float *q,
        uint32_t q_stride,
        const void *k_cache,
        const void *v_cache,
        const int32_t *page_indices,
        uint32_t n_pages,
        uint32_t page_size,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t base_seq_len,
        uint32_t batch,
        uint32_t q_batch_stride,
        uint32_t out_batch_stride,
        float scale,
        cudaStream_t stream,
        bool *plan_cache_hit_out) {
    unsigned threads = 0, blocks = 0;
    if (plan_cache_hit_out) *plan_cache_hit_out = false;
    const bool launched =
        run_batch_prefill_paged_typed<__nv_fp8_e4m3>(
            out, q_f16, o_f16, int_workspace, host_int_workspace,
            int_workspace_bytes, float_workspace, float_workspace_bytes,
            q, q_stride, k_cache, v_cache, page_indices,
            n_pages, page_size, n_heads, n_kv_heads, head_dim,
            base_seq_len, batch, q_batch_stride, out_batch_stride,
            scale, stream, threads, blocks, plan_cache_hit_out);
    if (!launched) return false;
    unpack_gate_kernel<half><<<blocks, threads, 0, stream>>>(
        out, o_f16, q, q_stride, batch, n_heads, head_dim,
        q_batch_stride, out_batch_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool launch_batch_prefill_paged_ragged_f16q_f16kv_gated(
        float *out,
        half *q_f16,
        half *o_f16,
        int32_t *int_workspace,
        void *host_int_workspace,
        size_t int_workspace_bytes,
        float *float_workspace,
        size_t float_workspace_bytes,
        const float *q,
        uint32_t q_stride,
        const void *k_cache,
        const void *v_cache,
        const int32_t *page_indices,
        const int32_t *page_indptr,
        const int32_t *last_page_len,
        const int32_t *q_indptr,
        const int32_t *q_indptr_host,
        const int32_t *page_indptr_host,
        uint32_t batch,
        uint32_t total_q,
        uint32_t page_size,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t q_batch_stride,
        uint32_t out_batch_stride,
        float scale,
        cudaStream_t stream) {
    unsigned threads = 0, blocks = 0;
    if (!run_batch_prefill_paged_ragged_typed<half>(
            out, q_f16, o_f16, int_workspace, host_int_workspace,
            int_workspace_bytes, float_workspace, float_workspace_bytes,
            q, q_stride, k_cache, v_cache,
            page_indices, page_indptr, last_page_len, q_indptr,
            q_indptr_host, page_indptr_host, batch, total_q, page_size,
            n_heads, n_kv_heads, head_dim, q_batch_stride, out_batch_stride,
            scale, stream, threads, blocks)) {
        return false;
    }
    unpack_gate_kernel<half><<<blocks, threads, 0, stream>>>(
        out, o_f16, q, q_stride, total_q, n_heads, head_dim,
        q_batch_stride, out_batch_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool launch_batch_prefill_paged_ragged_f16q_fp8kv_gated(
        float *out,
        half *q_f16,
        half *o_f16,
        int32_t *int_workspace,
        void *host_int_workspace,
        size_t int_workspace_bytes,
        float *float_workspace,
        size_t float_workspace_bytes,
        const float *q,
        uint32_t q_stride,
        const void *k_cache,
        const void *v_cache,
        const int32_t *page_indices,
        const int32_t *page_indptr,
        const int32_t *last_page_len,
        const int32_t *q_indptr,
        const int32_t *q_indptr_host,
        const int32_t *page_indptr_host,
        uint32_t batch,
        uint32_t total_q,
        uint32_t page_size,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t q_batch_stride,
        uint32_t out_batch_stride,
        float scale,
        cudaStream_t stream) {
    unsigned threads = 0, blocks = 0;
    if (!run_batch_prefill_paged_ragged_typed<__nv_fp8_e4m3>(
            out, q_f16, o_f16, int_workspace, host_int_workspace,
            int_workspace_bytes, float_workspace, float_workspace_bytes,
            q, q_stride, k_cache, v_cache,
            page_indices, page_indptr, last_page_len, q_indptr,
            q_indptr_host, page_indptr_host, batch, total_q, page_size,
            n_heads, n_kv_heads, head_dim, q_batch_stride, out_batch_stride,
            scale, stream, threads, blocks)) {
        return false;
    }
    unpack_gate_kernel<half><<<blocks, threads, 0, stream>>>(
        out, o_f16, q, q_stride, total_q, n_heads, head_dim,
        q_batch_stride, out_batch_stride);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace flashinfer_adapter
} // namespace qw3
