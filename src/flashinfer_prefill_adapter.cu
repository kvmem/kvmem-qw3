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
#include <limits>
#include <vector>

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

bool stable_fp8_prefill_enabled() {
    const char *raw = std::getenv("QW3_EXPERIMENTAL_FP8_STABLE_PREFILL");
    if (!raw || !*raw) return false;
    return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0 &&
           std::strcmp(raw, "off") != 0 && std::strcmp(raw, "no") != 0;
}

bool stable_ragged_verify_shape(const int32_t *q_indptr, uint32_t batch,
                               uint32_t page_size, uint32_t heads,
                               uint32_t kv_heads, uint32_t head_dim) {
    const char *raw = std::getenv("QW3_EXPERIMENTAL_FP8_STABLE_RAGGED_PREFILL");
    if (!raw || std::strcmp(raw, "1") != 0 || !q_indptr || batch == 0 ||
        heads != 24 || kv_heads != 4 || head_dim != 256 ||
        (page_size != 16 && page_size != 32)) return false;
    for (uint32_t i = 0; i < batch; ++i)
        if (q_indptr[i + 1] <= q_indptr[i] || q_indptr[i + 1] - q_indptr[i] > 8)
            return false;
    return true;
}

// Match the fixed eight-row FP8 verifier: one 64-row packed-Q tile per
// request and 1024-token KV partitions. The ordinary planner chooses both
// from the other requests in the batch, including switching to CTA_Q=16
// when a request reaches a one/two-token verification tail.
// Keep real Q lengths in the output/merge descriptors; padding the planner's
// Q indptr alone would create holes in another request's output.
cudaError_t plan_stable_ragged_verify(
        flashinfer::PrefillPlanInfo &info, void *int_workspace,
        void *host_workspace, size_t int_bytes,
        float *float_workspace, size_t float_bytes,
        const int32_t *q_indptr, const int32_t *page_indptr,
        uint32_t batch, uint32_t total_q, uint32_t page_size,
        cudaStream_t stream) {
    constexpr uint32_t heads = 24, dim = 256, tile_q = 64;
    const uint32_t chunk_pages = 1024 / page_size;
    std::vector<int32_t> requests, q_tiles, kv_tiles, output{0}, merge{0};
    bool split = false;
    for (uint32_t i = 0; i < batch; ++i) {
        const int64_t pages = int64_t(page_indptr[i + 1]) - page_indptr[i];
        const int64_t rows = int64_t(q_indptr[i + 1]) - q_indptr[i];
        if (pages <= 0 || rows <= 0 || rows > 8) return cudaErrorInvalidValue;
        const int64_t chunks = (pages + chunk_pages - 1) / chunk_pages;
        if (int64_t(output.back()) + rows * chunks > INT32_MAX ||
            requests.size() + chunks > INT32_MAX) return cudaErrorInvalidValue;
        split |= chunks > 1;
        for (int32_t k = 0; k < chunks; ++k) {
            requests.push_back(i); q_tiles.push_back(0); kv_tiles.push_back(k);
        }
        for (int64_t q = 0; q < rows; ++q) merge.push_back(merge.back() + chunks);
        output.push_back(merge.back());
    }
    if (merge.size() != uint64_t(total_q) + 1 || (split && !float_workspace))
        return cudaErrorInvalidValue;
    info = {};
    info.cta_tile_q = tile_q; info.total_num_rows = total_q;
    info.padded_batch_size = requests.size(); info.split_kv = split;
    flashinfer::AlignedAllocator integers(int_workspace, int_bytes);
    auto copy_vector = [&](const std::vector<int32_t> &values, const char *name) {
        const auto offset = integers.aligned_alloc_offset(values.size() * sizeof(int32_t), 16, name);
        std::memcpy(static_cast<char *>(host_workspace) + offset, values.data(), values.size() * sizeof(int32_t));
        return offset;
    };
    info.request_indices_offset = copy_vector(requests, "stable_ragged_requests");
    info.qo_tile_indices_offset = copy_vector(q_tiles, "stable_ragged_q_tiles");
    info.kv_tile_indices_offset = copy_vector(kv_tiles, "stable_ragged_kv_tiles");
    info.o_indptr_offset = copy_vector(output, "stable_ragged_output");
    info.kv_chunk_size_ptr_offset = integers.aligned_alloc_offset(sizeof(int32_t), 1, "stable_ragged_chunk");
    *reinterpret_cast<int32_t *>(static_cast<char *>(host_workspace) + info.kv_chunk_size_ptr_offset) = 1024;
    if (split) {
        flashinfer::AlignedAllocator floats(float_workspace, float_bytes);
        // Kernels write only real qo_len rows at o_indptr[request] +
        // qo_idx * chunks + chunk. Reserving a whole CTA_Q tile per KV
        // partition wastes up to 64x and exceeds the default scratch limit
        // at long C4, even though the actual merge payload fits comfortably.
        const size_t partial_rows = size_t(heads) * merge.back();
        info.v_offset = floats.aligned_alloc_offset(partial_rows * dim * sizeof(float), 16, "stable_ragged_v");
        info.s_offset = floats.aligned_alloc_offset(partial_rows * sizeof(float), 16, "stable_ragged_s");
        info.merge_indptr_offset = copy_vector(merge, "stable_ragged_merge");
        info.block_valid_mask_offset = integers.aligned_alloc_offset(requests.size() * sizeof(bool), 16, "stable_ragged_valid");
        std::fill_n(reinterpret_cast<bool *>(static_cast<char *>(host_workspace) + info.block_valid_mask_offset), requests.size(), true);
    }
    // Construct all host descriptors before the single asynchronous copy.
    return cudaMemcpyAsync(int_workspace, host_workspace, integers.num_allocated_bytes(),
                           cudaMemcpyHostToDevice, stream);
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
// shape. Cache the CPU scheduling result, NOT an assumption that a borrowed
// device workspace still contains that plan: ragged prefill, a different KV
// dtype, or a recycled backend allocation may have overwritten it. Restore
// cached metadata through the caller's fenced pinned slot on every hit.
struct PagedPrefillPlanCache {
    bool valid = false;
    int device = -1;
    std::vector<char> metadata;
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
    bool fixed_verify_schedule = false;
    uint64_t consecutive_hits = 0;
    uint64_t total_hits = 0;
    uint64_t total_misses = 0;
    flashinfer::PrefillPlanInfo plan_info;

    bool matches(
            int32_t *iw, size_t iw_bytes,
            float *fw, size_t fw_bytes, cudaStream_t s,
            uint32_t b, uint32_t pages, uint32_t nh,
            uint32_t nkh, uint32_t hd, uint32_t ps,
            bool allow_split, int current_device) const {
        return valid &&
               device == current_device &&
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
    // Diagnostic-only stable schedule for the Qwen FP8 verifier shape. Plan
    // enough Q rows for an eight-row call even for a serial call. Device Q
    // indptr and max_total_num_rows remain the actual row count, so no padded
    // input or output is accessed. Extra merge descriptors are unused.
    const bool fixed_verify_schedule = stable_fp8_prefill_enabled() && sizeof(TKV) == 1 && head_dim == 256 &&
        n_heads == 24 && n_kv_heads == 4 && (page_size == 16 || page_size == 32) && batch <= 8;
    const uint32_t planning_rows = fixed_verify_schedule ? 8u : batch;
    int32_t qo_indptr_h[2] = {0, static_cast<int32_t>(planning_rows)};
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
    if (fixed_verify_schedule && need_pages > (1024 / page_size) && !allow_split_kv) return false;
    float *active_float_workspace = float_workspace;
    size_t active_float_workspace_bytes = float_workspace_bytes;
    static thread_local PagedPrefillPlanCache plan_cache;
    const bool use_plan_cache = paged_prefill_plan_cache_enabled();
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) return false;
    const bool plan_cache_hit =
        use_plan_cache &&
        plan_cache.fixed_verify_schedule == fixed_verify_schedule &&
        plan_cache.matches(
            int_workspace, int_workspace_bytes,
            float_workspace, float_workspace_bytes, stream,
            batch, need_pages, n_heads, n_kv_heads, head_dim,
            page_size, allow_split_kv, current_device);
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
            planning_rows,
            1,
            n_heads,
            n_kv_heads,
            head_dim,
            head_dim,
            page_size,
            false,
            sizeof(half),
            -1,
            fixed_verify_schedule ? (1024 / page_size) : 0, // 1024 KV tokens, independent of Q width
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
        std::memcpy(plan_host_workspace, plan_cache.metadata.data(),
                    plan_cache.metadata.size());
        if (cudaMemcpyAsync(plan_int_workspace, plan_host_workspace,
                            plan_cache.metadata.size(), cudaMemcpyHostToDevice,
                            stream) != cudaSuccess) return false;
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
        // A failed plan may partially overwrite the borrowed workspaces.
        plan_cache.valid = false;
        cudaError_t plan_st = cudaErrorInvalidValue;
        try {
            plan_st = plan(/*disable_split_kv=*/!allow_split_kv);
        } catch (const std::exception &) {
            // Do not hide an undersized fixed schedule with a different algorithm.
            if (fixed_verify_schedule) return false;
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
            // These are the final allocations in non-graph PrefillPlan (also
            // mirrored by prefill_plan_workspace_size below). Copy only the
            // live metadata extent, not the potentially multi-MiB capacity.
            const size_t metadata_bytes = plan_info.split_kv
                ? size_t(plan_info.block_valid_mask_offset) +
                    size_t(plan_info.padded_batch_size) * sizeof(bool)
                : size_t(plan_info.kv_chunk_size_ptr_offset) + sizeof(int32_t);
            if (metadata_bytes > plan_int_workspace_bytes) return false;
            const auto *metadata = static_cast<const char *>(plan_host_workspace);
            plan_cache.metadata.assign(metadata, metadata + metadata_bytes);
            plan_cache.device = current_device;
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
            plan_cache.fixed_verify_schedule = fixed_verify_schedule;
            plan_cache.plan_info = plan_info;
            plan_cache.valid = true;
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
    const bool fixed_verify_schedule = sizeof(TKV) == 1 &&
        stable_ragged_verify_shape(q_indptr_host, batch, page_size, n_heads, n_kv_heads, head_dim);
    auto plan = [&](bool disable_split_kv) {
        if (fixed_verify_schedule)
            return plan_stable_ragged_verify(plan_info, int_workspace, host_int_workspace,
                int_workspace_bytes, float_workspace, float_workspace_bytes,
                q_indptr_host, page_indptr_host, batch, total_q, page_size, stream);
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
        // A fixed schedule must not silently change arithmetic on allocation
        // failure. Its workspace is sized together with the ordinary plan.
        if (fixed_verify_schedule) return false;
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

bool checked_workspace_alloc(size_t &offset,
                             size_t count,
                             size_t element_bytes,
                             size_t alignment) {
    if (alignment == 0 || count > std::numeric_limits<size_t>::max() / element_bytes) {
        return false;
    }
    const size_t bytes = count * element_bytes;
    const size_t padding = (alignment - (offset % alignment)) % alignment;
    if (offset > std::numeric_limits<size_t>::max() - padding ||
        offset + padding > std::numeric_limits<size_t>::max() - bytes) {
        return false;
    }
    offset += padding + bytes;
    return true;
}

bool checked_workspace_mul(size_t lhs, size_t rhs, size_t &result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

// FlashInfer's planner exposes the scheduling primitive used by PrefillPlan,
// but older compatible headers do not yet expose PrefillPlanWorkspaceSize.
// Mirror PrefillPlan's allocation sequence so workspace sizing works with both
// header revisions while remaining exact for the plan that will be launched.
cudaError_t prefill_plan_workspace_size(
        size_t &float_workspace_bytes,
        size_t &int_workspace_bytes,
        int32_t *qo_indptr_h,
        int32_t *kv_indptr_h,
        uint32_t total_num_rows,
        uint32_t batch_size,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t page_size, int32_t fixed_split_size = 0) {
    if (!qo_indptr_h || !kv_indptr_h || batch_size == 0 ||
        n_kv_heads == 0 || n_heads % n_kv_heads != 0) {
        return cudaErrorInvalidValue;
    }

    int device = 0;
    int num_sms = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    status = cudaDeviceGetAttribute(
        &num_sms, cudaDevAttrMultiProcessorCount, device);
    if (status != cudaSuccess) return status;

    const int64_t available_ctas =
        std::max<int64_t>(0, static_cast<int64_t>(num_sms) * 2);
    const uint32_t max_batch_size_if_split = static_cast<uint32_t>(
        available_ctas / static_cast<int64_t>(n_kv_heads));

    try {
        auto split_plan = flashinfer::PrefillSplitQOKVIndptr<int32_t>(
            qo_indptr_h,
            kv_indptr_h,
            total_num_rows,
            batch_size,
            n_heads,
            n_kv_heads,
            head_dim,
            page_size,
            max_batch_size_if_split,
            /*enable_cuda_graph=*/false,
            /*window_left=*/-1,
            /*fixed_split_size=*/fixed_split_size,
            /*disable_split_kv=*/false);

        const bool split_kv = std::get<0>(split_plan);
        const size_t padded_batch_size = std::get<2>(split_plan);
        const size_t cta_tile_q = std::get<3>(split_plan);

        size_t int_bytes = 0;
        if (!checked_workspace_alloc(
                int_bytes, padded_batch_size, sizeof(int32_t), 16) ||
            !checked_workspace_alloc(
                int_bytes, padded_batch_size, sizeof(int32_t), 16) ||
            !checked_workspace_alloc(
                int_bytes, padded_batch_size, sizeof(int32_t), 16) ||
            !checked_workspace_alloc(
                int_bytes, static_cast<size_t>(batch_size) + 1,
                sizeof(int32_t), 16) ||
            !checked_workspace_alloc(
                int_bytes, 1, sizeof(int32_t), 1)) {
            return cudaErrorInvalidValue;
        }

        size_t float_bytes = 0;
        if (split_kv) {
            size_t partial_rows = 0;
            size_t partial_values = 0;
            if (!checked_workspace_mul(
                    n_heads, padded_batch_size, partial_rows) ||
                !checked_workspace_mul(
                    partial_rows, cta_tile_q, partial_rows) ||
                !checked_workspace_mul(
                    partial_rows, head_dim, partial_values) ||
                !checked_workspace_alloc(
                    float_bytes, partial_values, sizeof(float), 16) ||
                !checked_workspace_alloc(
                    float_bytes, partial_rows, sizeof(float), 16) ||
                !checked_workspace_alloc(
                    int_bytes, static_cast<size_t>(total_num_rows) + 1,
                    sizeof(int32_t), 16) ||
                !checked_workspace_alloc(
                    int_bytes, padded_batch_size, sizeof(bool), 16)) {
                return cudaErrorInvalidValue;
            }
        }

        int_workspace_bytes = int_bytes;
        float_workspace_bytes = float_bytes;
        return cudaSuccess;
    } catch (const std::exception &) {
        return cudaErrorInvalidValue;
    }
}

} // namespace

bool batch_prefill_paged_workspace_bytes(
        size_t &int_workspace_bytes,
        size_t &float_workspace_bytes,
        uint32_t n_pages,
        uint32_t page_size,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t base_seq_len,
        uint32_t batch) {
    int_workspace_bytes = 0;
    float_workspace_bytes = 0;
    if (batch == 0 || page_size == 0 || n_heads == 0 ||
        n_kv_heads == 0 || n_heads % n_kv_heads != 0) {
        return false;
    }
    const uint32_t kv_len = base_seq_len + batch;
    const uint32_t need_pages =
        std::max<uint32_t>((kv_len + page_size - 1U) / page_size, 1U);
    if (need_pages > n_pages ||
        batch > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
        need_pages >
            static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        return false;
    }

    int32_t qo_indptr_h[2] = {0, static_cast<int32_t>(batch)};
    int32_t kv_indptr_h[2] = {0, static_cast<int32_t>(need_pages)};
    size_t plan_int_bytes = 0;
    cudaError_t status =
        prefill_plan_workspace_size(
            float_workspace_bytes,
            plan_int_bytes,
            qo_indptr_h,
            kv_indptr_h,
            batch,
            1,
            n_heads,
            n_kv_heads,
            head_dim,
            /*page_size=*/page_size);
    // This dtype-independent capacity query must accommodate both the
    // original plan and the FP8 fixed plan. Keep the original capacity for
    // FP16 callers. Ragged workspace sizing handles its optional fixed plan
    // separately, using the same 1024-token partition width.
    if (stable_fp8_prefill_enabled() && status == cudaSuccess && head_dim == 256 && n_heads == 24 &&
        n_kv_heads == 4 && (page_size == 16 || page_size == 32) && batch <= 8) {
        int32_t fixed_qo[2] = {0, 8};
        size_t fixed_float_bytes = 0, fixed_int_bytes = 0;
        status = prefill_plan_workspace_size(
            fixed_float_bytes, fixed_int_bytes, fixed_qo, kv_indptr_h,
            8, 1, n_heads, n_kv_heads, head_dim, page_size, 1024 / page_size);
        if (status == cudaSuccess) {
            float_workspace_bytes = std::max(float_workspace_bytes, fixed_float_bytes);
            plan_int_bytes = std::max(plan_int_bytes, fixed_int_bytes);
        }
    }
    if (status != cudaSuccess ||
        plan_int_bytes >
            std::numeric_limits<size_t>::max() -
                kPagedPrefillPrefixI32 * sizeof(int32_t)) {
        int_workspace_bytes = 0;
        float_workspace_bytes = 0;
        return false;
    }
    int_workspace_bytes =
        kPagedPrefillPrefixI32 * sizeof(int32_t) + plan_int_bytes;
    return true;
}

bool batch_prefill_paged_ragged_workspace_bytes(
        size_t &int_workspace_bytes,
        size_t &float_workspace_bytes,
        const int32_t *q_indptr_host,
        const int32_t *page_indptr_host,
        uint32_t batch,
        uint32_t total_q,
        uint32_t page_size,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim) {
    int_workspace_bytes = 0;
    float_workspace_bytes = 0;
    if (!q_indptr_host || !page_indptr_host || batch == 0 ||
        total_q == 0 || page_size == 0 || n_heads == 0 ||
        n_kv_heads == 0 || n_heads % n_kv_heads != 0 ||
        q_indptr_host[0] != 0 ||
        q_indptr_host[batch] != static_cast<int32_t>(total_q) ||
        page_indptr_host[0] != 0 || page_indptr_host[batch] <= 0) {
        return false;
    }
    cudaError_t status =
        prefill_plan_workspace_size(
            float_workspace_bytes,
            int_workspace_bytes,
            const_cast<int32_t *>(q_indptr_host),
            const_cast<int32_t *>(page_indptr_host),
            total_q,
            batch,
            n_heads,
            n_kv_heads,
            head_dim,
            page_size);
    if (status == cudaSuccess && stable_ragged_verify_shape(
            q_indptr_host, batch, page_size, n_heads, n_kv_heads, head_dim)) {
        // Eight planning rows force CTA_Q=64. This overallocates only merge
        // metadata; execution uses real per-request row counts throughout.
        if (batch > uint32_t(INT32_MAX / 8)) return false;
        std::vector<int32_t> padded_q(batch + 1);
        for (uint32_t i = 0; i <= batch; ++i) padded_q[i] = i * 8;
        size_t fixed_float_bytes = 0, fixed_int_bytes = 0;
        status = prefill_plan_workspace_size(fixed_float_bytes, fixed_int_bytes,
            padded_q.data(), const_cast<int32_t *>(page_indptr_host), batch * 8,
            batch, n_heads, n_kv_heads, head_dim, page_size, 1024 / page_size);
        if (status == cudaSuccess) {
            size_t partial_rows = 0;
            bool split = false;
            for (uint32_t i = 0; i < batch; ++i) {
                const size_t chunks = (size_t(page_indptr_host[i + 1] - page_indptr_host[i]) +
                                       1024 / page_size - 1) / (1024 / page_size);
                partial_rows += size_t(q_indptr_host[i + 1] - q_indptr_host[i]) * chunks;
                split |= chunks > 1;
            }
            // Same compact layout as plan_stable_ragged_verify. Keep the
            // ordinary plan's requirement too for FP16/unsupported callers.
            fixed_float_bytes = split ? partial_rows * n_heads * (head_dim + 1) * sizeof(float) : 0;
            float_workspace_bytes = std::max(float_workspace_bytes, fixed_float_bytes);
            int_workspace_bytes = std::max(int_workspace_bytes, fixed_int_bytes);
        }
    }
    if (status != cudaSuccess) {
        int_workspace_bytes = 0;
        float_workspace_bytes = 0;
        return false;
    }
    return true;
}

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
