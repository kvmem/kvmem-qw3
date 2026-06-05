#include "flashinfer_decode_adapter.hpp"

#include <flashinfer/attention/cascade.cuh>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/pos_enc.cuh>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace qw3 {
namespace flashinfer_adapter {
namespace {

uint32_t ceil_div_u32(uint32_t x, uint32_t y) {
    return (x + y - 1U) / y;
}

bool batch_decode_multirow_merge_enabled() {
    static const bool enabled = []() {
        const char *raw = std::getenv("QW3_FLASHINFER_BATCH_DECODE_MULTIROW_MERGE");
        if (raw == nullptr || *raw == '\0') {
            raw = std::getenv("QW3_EXPERIMENTAL_FLASHINFER_BATCH_DECODE_MULTIROW_MERGE");
        }
        if (raw == nullptr || *raw == '\0') return true;
        return std::strcmp(raw, "0") != 0 &&
               std::strcmp(raw, "false") != 0 &&
               std::strcmp(raw, "off") != 0 &&
               std::strcmp(raw, "no") != 0;
    }();
    return enabled;
}

__global__ void unpack_o_kernel(float *out,
                                const half *o_f16,
                                uint64_t elems) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elems) return;
    out[i] = __half2float(o_f16[i]);
}

__global__ void unpack_o_batch_kernel(float *out,
                                      const half *o_f16,
                                      uint32_t batch,
                                      uint32_t n_heads,
                                      uint32_t head_dim,
                                      uint32_t out_batch_stride) {
    const uint64_t total = static_cast<uint64_t>(batch) * n_heads * head_dim;
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;

    const uint64_t row_elems = static_cast<uint64_t>(n_heads) * head_dim;
    const uint32_t b = static_cast<uint32_t>(i / row_elems);
    const uint64_t inner = i - static_cast<uint64_t>(b) * row_elems;
    out[static_cast<uint64_t>(b) * out_batch_stride + inner] = __half2float(o_f16[i]);
}

__global__ void unpack_gate_batch_kernel(float *out,
                                         const half *o_f16,
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
    const float value = __half2float(o_f16[i]);
    const float gate = q[q_off + head_dim + d];
    out[static_cast<uint64_t>(b) * out_batch_stride
        + static_cast<uint64_t>(h) * head_dim + d] =
        value * (1.0f / (1.0f + expf(-gate)));
}

constexpr uint32_t decode_num_threads(uint32_t group_size, uint32_t sizeof_dtype, uint32_t bdx) {
    const uint32_t heuristic =
        (group_size == 8U) ? ((sizeof_dtype == 1U) ? 256U : 512U) : 128U;
    return heuristic > bdx * group_size ? heuristic : bdx * group_size;
}

template <uint32_t HeadDim,
          uint32_t GroupSize,
          uint32_t NumStages,
          typename Params,
          typename Variant>
cudaError_t launch_decode_group_size(Params params,
                                     typename Params::DTypeO *tmp,
                                     cudaStream_t stream) {
    using DTypeKV = typename Params::DTypeKV;
    constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HeadDim / 32UL);
    constexpr uint32_t bdx = HeadDim / vec_size;
    constexpr uint32_t bdy = GroupSize;
    constexpr uint32_t num_threads = decode_num_threads(GroupSize, sizeof(DTypeKV), bdx);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx =
        GroupSize == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 8U) : 1U;
    constexpr uint32_t smem_size =
        2U * NumStages * bdy * tile_size_per_bdx * bdz * HeadDim * sizeof(DTypeKV) +
        2U * bdy * bdz * sizeof(float);

    auto kernel = flashinfer::SingleDecodeWithKVCacheKernel<
        flashinfer::PosEncodingMode::kNone,
        NumStages,
        tile_size_per_bdx,
        vec_size,
        bdx,
        bdy,
        bdz,
        Variant,
        Params>;
    cudaError_t st = cudaFuncSetAttribute(kernel,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          smem_size);
    if (st != cudaSuccess) return st;

    const uint32_t seq_len = params.kv_len;
    const uint32_t num_qo_heads = params.num_qo_heads;
    const uint32_t num_kv_heads = params.num_kv_heads;
    if (seq_len <= 256 || tmp == nullptr) {
        dim3 nblks(1, num_kv_heads);
        dim3 nthrs(bdx, bdy, bdz);
        params.kv_chunk_size = seq_len;
        void *args[] = {static_cast<void *>(&params)};
        return cudaLaunchKernel(reinterpret_cast<void *>(kernel), nblks, nthrs,
                                args, smem_size, stream);
    }

    int dev_id = 0;
    int num_sm = 0;
    int num_blocks_per_sm = 0;
    st = cudaGetDevice(&dev_id);
    if (st != cudaSuccess) return st;
    st = cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id);
    if (st != cudaSuccess) return st;
    st = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm,
                                                       kernel,
                                                       num_threads,
                                                       smem_size);
    if (st != cudaSuccess) return st;
    const uint32_t max_grid_size = static_cast<uint32_t>(num_blocks_per_sm) *
                                   static_cast<uint32_t>(num_sm);
    const uint32_t max_num_kv_chunks = std::max(max_grid_size / num_kv_heads, 1U);
    const uint32_t kv_chunk_size = std::max(ceil_div_u32(seq_len, max_num_kv_chunks), 256U);
    const uint32_t num_chunks = ceil_div_u32(seq_len, kv_chunk_size);
    dim3 nblks(num_chunks, num_kv_heads);
    dim3 nthrs(bdx, bdy, bdz);

    auto *tmp_lse = reinterpret_cast<float *>(tmp + static_cast<uint64_t>(num_chunks) *
                                                    num_qo_heads * HeadDim);
    auto *o = params.o;
    auto *lse = params.lse;
    params.o = tmp;
    params.lse = tmp_lse;
    params.kv_chunk_size = kv_chunk_size;
    void *args[] = {static_cast<void *>(&params)};
    st = cudaLaunchKernel(reinterpret_cast<void *>(kernel), nblks, nthrs,
                          args, smem_size, stream);
    if (st != cudaSuccess) return st;
    return flashinfer::MergeStates(tmp, tmp_lse, o, lse, num_chunks, 1,
                                   num_qo_heads, HeadDim, stream);
}

template <uint32_t HeadDim, uint32_t GroupSize, typename Params, typename Variant>
cudaError_t launch_decode_group_size_dispatch_stages(Params params,
                                                    typename Params::DTypeO *tmp,
                                                    cudaStream_t stream) {
    const auto cc = flashinfer::GetCudaComputeCapability();
    if (cc.first >= 8) {
        return launch_decode_group_size<HeadDim, GroupSize, 2, Params, Variant>(
            params, tmp, stream);
    }
    return launch_decode_group_size<HeadDim, GroupSize, 1, Params, Variant>(
        params, tmp, stream);
}

template <uint32_t HeadDim, typename DTypeKV>
bool launch_decode_head_dim(float *out,
                            half *o_f16,
                            half *tmp,
                            const float *q,
                            uint32_t q_stride,
                            const void *k_cache,
                            const void *v_cache,
                            uint32_t n_heads,
                            uint32_t n_kv_heads,
                            uint32_t seq_len,
                            float scale,
                            cudaStream_t stream) {
    using Params = flashinfer::SingleDecodeParams<float, DTypeKV, half>;
    using Variant = flashinfer::DefaultAttention<false, false, false, false>;

    Params params;
    params.q = const_cast<float *>(q);
    params.k = const_cast<DTypeKV *>(static_cast<const DTypeKV *>(k_cache));
    params.v = const_cast<DTypeKV *>(static_cast<const DTypeKV *>(v_cache));
    params.o = o_f16;
    params.lse = nullptr;
    params.maybe_alibi_slopes = nullptr;
    params.kv_len = seq_len;
    params.num_qo_heads = n_heads;
    params.num_kv_heads = n_kv_heads;
    params.q_stride_n = n_heads * q_stride;
    params.q_stride_h = q_stride;
    params.kv_stride_n = n_kv_heads * HeadDim;
    params.kv_stride_h = HeadDim;
    params.window_left = -1;
    params.logits_soft_cap = 0.0f;
    params.sm_scale = scale;
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;
    params.kv_chunk_size = 0;

    const uint32_t group_size = n_heads / n_kv_heads;
    cudaError_t st = cudaSuccess;
    if (group_size == 6) {
        st = launch_decode_group_size_dispatch_stages<HeadDim, 6, Params, Variant>(
            params, tmp, stream);
    } else {
        st = flashinfer::SingleDecodeWithKVCacheDispatched<
            HeadDim,
            flashinfer::PosEncodingMode::kNone,
            Variant>(params, tmp, stream);
    }
    if (st != cudaSuccess) return false;

    const uint64_t elems = static_cast<uint64_t>(n_heads) * HeadDim;
    const unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((elems + threads - 1) / threads);
    unpack_o_kernel<<<blocks, threads, 0, stream>>>(out, o_f16, elems);
    return cudaGetLastError() == cudaSuccess;
}

template <uint32_t HeadDim,
          uint32_t GroupSize,
          uint32_t NumStages,
          typename Params,
          typename Variant>
uint32_t plan_batch_decode_chunk_size(uint32_t n_kv_heads,
                                      uint32_t batch,
                                      uint32_t max_seq_len) {
    using DTypeKV = typename Params::DTypeKV;
    constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HeadDim / 32UL);
    constexpr uint32_t bdx = HeadDim / vec_size;
    constexpr uint32_t bdy = GroupSize;
    constexpr uint32_t num_threads = decode_num_threads(GroupSize, sizeof(DTypeKV), bdx);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx =
        GroupSize == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 8U) : 1U;
    constexpr uint32_t smem_size =
        2U * NumStages * bdy * tile_size_per_bdx * bdz * HeadDim * sizeof(DTypeKV) +
        2U * bdy * bdz * sizeof(float);

    auto kernel = flashinfer::BatchDecodeWithPagedKVCacheKernel<
        flashinfer::PosEncodingMode::kNone,
        NumStages,
        tile_size_per_bdx,
        vec_size,
        bdx,
        bdy,
        bdz,
        Variant,
        Params>;

    int dev_id = 0;
    int num_sm = 0;
    int num_blocks_per_sm = 0;
    cudaError_t st = cudaGetDevice(&dev_id);
    if (st != cudaSuccess) return 0;
    st = cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id);
    if (st != cudaSuccess) return 0;
    st = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm,
                                                       kernel,
                                                       num_threads,
                                                       smem_size);
    if (st != cudaSuccess || num_sm <= 0 || num_blocks_per_sm <= 0) return 0;
    const uint32_t max_grid = static_cast<uint32_t>(num_sm) *
                              static_cast<uint32_t>(num_blocks_per_sm);
    (void)batch;
    // Match the row-wise SingleDecode partitioning scale. Keeping the same KV
    // chunk size is more important than minimizing batch-decode grid size
    // because MTP verifier acceptance is greedy-token sensitive.
    const uint32_t denom = std::max<uint32_t>(n_kv_heads, 1U);
    const uint32_t max_chunks = std::max<uint32_t>(max_grid / denom, 1U);
    return std::max<uint32_t>(ceil_div_u32(max_seq_len, max_chunks), 256U);
}

template <uint32_t HeadDim,
          uint32_t GroupSize,
          uint32_t NumStages,
          typename Params,
          typename Variant>
cudaError_t launch_batch_decode_group_size(Params params,
                                           typename Params::DTypeO *tmp_v,
                                           float *tmp_s,
                                           const int32_t *o_indptr_host,
                                           bool partition_kv,
                                           bool rowwise_merge,
                                           cudaStream_t stream) {
    using DTypeKV = typename Params::DTypeKV;
    constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HeadDim / 32UL);
    constexpr uint32_t bdx = HeadDim / vec_size;
    constexpr uint32_t bdy = GroupSize;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GroupSize == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 4U) : 1U;
    constexpr uint32_t smem_size =
        2U * NumStages * tile_size_per_bdx * bdy * bdz * HeadDim * sizeof(DTypeKV) +
        std::max(tile_size_per_bdx * num_threads * sizeof(DTypeKV*),
                 2U * bdy * bdz * sizeof(float));

    auto kernel = flashinfer::BatchDecodeWithPagedKVCacheKernel<
        flashinfer::PosEncodingMode::kNone,
        NumStages,
        tile_size_per_bdx,
        vec_size,
        bdx,
        bdy,
        bdz,
        Variant,
        Params>;
    cudaError_t st = cudaFuncSetAttribute(kernel,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          smem_size);
    if (st != cudaSuccess) return st;

    dim3 nblks(params.padded_batch_size, params.paged_kv.num_heads);
    dim3 nthrs(bdx, bdy, bdz);
    if (!partition_kv) {
        params.partition_kv = false;
        void *args[] = {static_cast<void *>(&params)};
        return cudaLaunchKernel(reinterpret_cast<void *>(kernel), nblks, nthrs,
                                args, smem_size, stream);
    }
    if (tmp_v == nullptr || tmp_s == nullptr) return cudaErrorInvalidValue;

    params.partition_kv = true;
    auto *o = params.o;
    auto *lse = params.lse;
    params.o = tmp_v;
    params.lse = tmp_s;
    void *args[] = {static_cast<void *>(&params)};
    st = cudaLaunchKernel(reinterpret_cast<void *>(kernel), nblks, nthrs,
                          args, smem_size, stream);
    if (st != cudaSuccess) return st;
    if (rowwise_merge) {
        if (o_indptr_host == nullptr) return cudaErrorInvalidValue;
        if (batch_decode_multirow_merge_enabled() &&
            params.paged_kv.batch_size > 1 &&
            o_indptr_host[0] == 0) {
            const uint32_t tiles_per_row =
                static_cast<uint32_t>(o_indptr_host[1] - o_indptr_host[0]);
            bool uniform = tiles_per_row > 0;
            for (uint32_t b = 1; uniform && b < params.paged_kv.batch_size; ++b) {
                const uint32_t begin = static_cast<uint32_t>(o_indptr_host[b]);
                const uint32_t end = static_cast<uint32_t>(o_indptr_host[b + 1]);
                uniform = (end - begin) == tiles_per_row &&
                          begin == b * tiles_per_row;
            }
            if (uniform) {
                // Preserve FlashInfer's MergeStates numerics while reducing
                // one merge launch per verifier row to one launch per batch.
                return flashinfer::MergeStates(tmp_v, tmp_s, o, lse,
                                               tiles_per_row,
                                               params.paged_kv.batch_size,
                                               params.num_qo_heads, HeadDim,
                                               stream);
            }
        }
        for (uint32_t b = 0; b < params.paged_kv.batch_size; ++b) {
            const uint32_t begin = static_cast<uint32_t>(o_indptr_host[b]);
            const uint32_t end = static_cast<uint32_t>(o_indptr_host[b + 1]);
            if (end <= begin) return cudaErrorInvalidValue;
            st = flashinfer::MergeStates(
                tmp_v + static_cast<uint64_t>(begin) * params.num_qo_heads * HeadDim,
                tmp_s + static_cast<uint64_t>(begin) * params.num_qo_heads,
                o + static_cast<uint64_t>(b) * params.num_qo_heads * HeadDim,
                lse == nullptr ? nullptr : lse + static_cast<uint64_t>(b) * params.num_qo_heads,
                end - begin, 1, params.num_qo_heads, HeadDim, stream);
            if (st != cudaSuccess) return st;
        }
        return cudaSuccess;
    }
    return flashinfer::VariableLengthMergeStates(
        tmp_v, tmp_s, params.o_indptr, o, lse,
        params.paged_kv.batch_size, nullptr, params.num_qo_heads,
        HeadDim, /*enable_pdl=*/false, stream);
}

template <uint32_t HeadDim, typename Params, typename Variant>
uint32_t plan_batch_decode_head_dim(uint32_t n_heads,
                                    uint32_t n_kv_heads,
                                    uint32_t batch,
                                    uint32_t max_seq_len) {
    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return 0;
    const uint32_t group_size = n_heads / n_kv_heads;
    const auto cc = flashinfer::GetCudaComputeCapability();
    if (cc.first < 8) return 0;
    if (group_size == 6) {
        return plan_batch_decode_chunk_size<HeadDim, 6, 2, Params, Variant>(
            n_kv_heads, batch, max_seq_len);
    }
    return 0;
}

template <uint32_t HeadDim, typename Params, typename Variant>
cudaError_t launch_batch_decode_head_dim_dispatch(Params params,
                                                  typename Params::DTypeO *tmp_v,
                                                  float *tmp_s,
                                                  const int32_t *o_indptr_host,
                                                  bool partition_kv,
                                                  bool rowwise_merge,
                                                  cudaStream_t stream) {
    const uint32_t n_heads = params.num_qo_heads;
    const uint32_t n_kv_heads = params.paged_kv.num_heads;
    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return cudaErrorInvalidValue;
    const uint32_t group_size = n_heads / n_kv_heads;
    const auto cc = flashinfer::GetCudaComputeCapability();
    if (cc.first < 8) return cudaErrorInvalidDeviceFunction;
    if (group_size == 6) {
        return launch_batch_decode_group_size<HeadDim, 6, 2, Params, Variant>(
            params, tmp_v, tmp_s, o_indptr_host, partition_kv, rowwise_merge, stream);
    }
    return cudaErrorInvalidValue;
}

} // namespace

uint64_t decode_f32q_f16kv_tmp_elements(uint32_t n_heads,
                                        uint32_t head_dim,
                                        uint32_t seq_len) {
    if (seq_len <= 256 || n_heads == 0 || head_dim == 0) return 0;
    const uint32_t max_chunks = ceil_div_u32(seq_len, 256);
    const uint64_t partial_o = static_cast<uint64_t>(max_chunks) * n_heads * head_dim;
    const uint64_t partial_lse = static_cast<uint64_t>(max_chunks) * n_heads;
    const uint64_t partial_lse_f16_slots =
        (partial_lse * sizeof(float) + sizeof(half) - 1U) / sizeof(half);
    return partial_o + partial_lse_f16_slots;
}

bool launch_decode_f32q_f16kv(float *out,
                              half *o_f16,
                              half *tmp,
                              const float *q,
                              uint32_t q_stride,
                              const void *k_cache,
                              const void *v_cache,
                              uint32_t n_heads,
                              uint32_t n_kv_heads,
                              uint32_t head_dim,
                              uint32_t seq_len,
                              float scale,
                              cudaStream_t stream) {
    if (out == nullptr || o_f16 == nullptr || q == nullptr || k_cache == nullptr || v_cache == nullptr) {
        return false;
    }
    if (n_heads == 0 || n_kv_heads == 0 || (n_heads % n_kv_heads) != 0) return false;
    if (seq_len == 0 || q_stride < head_dim) return false;

    if (head_dim == 128) {
        return launch_decode_head_dim<128, half>(out, o_f16, tmp, q, q_stride, k_cache, v_cache,
                                           n_heads, n_kv_heads, seq_len, scale, stream);
    }
    if (head_dim == 256) {
        return launch_decode_head_dim<256, half>(out, o_f16, tmp, q, q_stride, k_cache, v_cache,
                                           n_heads, n_kv_heads, seq_len, scale, stream);
    }
    return false;
}

bool launch_decode_f32q_fp8kv(float *out,
                              half *o_f16,
                              half *tmp,
                              const float *q,
                              uint32_t q_stride,
                              const void *k_cache,
                              const void *v_cache,
                              uint32_t n_heads,
                              uint32_t n_kv_heads,
                              uint32_t head_dim,
                              uint32_t seq_len,
                              float scale,
                              cudaStream_t stream) {
    if (out == nullptr || o_f16 == nullptr || q == nullptr || k_cache == nullptr || v_cache == nullptr) {
        return false;
    }
    if (n_heads == 0 || n_kv_heads == 0 || (n_heads % n_kv_heads) != 0) return false;
    if (seq_len == 0 || q_stride < head_dim) return false;

    if (head_dim == 128) {
        return launch_decode_head_dim<128, __nv_fp8_e4m3>(out, o_f16, tmp, q, q_stride, k_cache, v_cache,
                                           n_heads, n_kv_heads, seq_len, scale, stream);
    }
    if (head_dim == 256) {
        return launch_decode_head_dim<256, __nv_fp8_e4m3>(out, o_f16, tmp, q, q_stride, k_cache, v_cache,
                                           n_heads, n_kv_heads, seq_len, scale, stream);
    }
    return false;
}

uint32_t batch_decode_f32q_f16kv_chunk_size(uint32_t n_heads,
                                            uint32_t n_kv_heads,
                                            uint32_t head_dim,
                                            uint32_t batch,
                                            uint32_t max_seq_len) {
    if (n_heads == 0 || n_kv_heads == 0 || batch == 0 || max_seq_len == 0) return 0;
    if ((n_heads % n_kv_heads) != 0) return 0;
    using Params = flashinfer::BatchDecodeParams<float, half, half, int32_t>;
    using Variant = flashinfer::DefaultAttention<false, false, false, false>;
    if (head_dim == 128) {
        return plan_batch_decode_head_dim<128, Params, Variant>(
            n_heads, n_kv_heads, batch, max_seq_len);
    }
    if (head_dim == 256) {
        return plan_batch_decode_head_dim<256, Params, Variant>(
            n_heads, n_kv_heads, batch, max_seq_len);
    }
    return 0;
}

bool launch_batch_decode_f32q_f16kv_gated(
        float *out,
        half *o_f16,
        half *tmp_v,
        float *tmp_s,
        const float *q,
        uint32_t q_stride,
        const void *k_cache,
        const void *v_cache,
        const int32_t *page_indices,
        const int32_t *page_indptr,
        const int32_t *last_page_len,
        const int32_t *request_indices,
        const int32_t *kv_tile_indices,
        const int32_t *o_indptr,
        const int32_t *o_indptr_host,
        const int32_t *kv_chunk_size,
        bool partition_kv,
        bool rowwise_merge,
        uint32_t page_size,
        uint32_t total_tiles,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t batch,
        uint32_t q_batch_stride,
        uint32_t out_batch_stride,
        float scale,
        cudaStream_t stream) {
    if (out == nullptr || o_f16 == nullptr || q == nullptr ||
        k_cache == nullptr || v_cache == nullptr || page_indices == nullptr ||
        page_indptr == nullptr || last_page_len == nullptr ||
        request_indices == nullptr || kv_tile_indices == nullptr ||
        o_indptr == nullptr || o_indptr_host == nullptr ||
        kv_chunk_size == nullptr) {
        return false;
    }
    if (partition_kv && (tmp_v == nullptr || tmp_s == nullptr)) {
        return false;
    }
    if (n_heads == 0 || n_kv_heads == 0 || (n_heads % n_kv_heads) != 0) return false;
    if (batch == 0 || total_tiles == 0 || page_size == 0 || q_stride < head_dim) return false;

    using Params = flashinfer::BatchDecodeParams<float, half, half, int32_t>;
    using Variant = flashinfer::DefaultAttention<false, false, false, false>;

    auto paged_kv = flashinfer::paged_kv_t<half, int32_t>(
        n_kv_heads, page_size, head_dim, batch, flashinfer::QKVLayout::kNHD,
        const_cast<half *>(static_cast<const half *>(k_cache)),
        const_cast<half *>(static_cast<const half *>(v_cache)),
        const_cast<int32_t *>(page_indices),
        const_cast<int32_t *>(page_indptr),
        const_cast<int32_t *>(last_page_len),
        nullptr);
    Params params(
        const_cast<float *>(q),
        nullptr,
        paged_kv,
        o_f16,
        nullptr,
        nullptr,
        n_heads,
        static_cast<int32_t>(q_batch_stride),
        static_cast<int32_t>(q_stride),
        -1,
        0.0f,
        scale,
        1.0f,
        1.0f);
    params.padded_batch_size = total_tiles;
    params.request_indices = const_cast<int32_t *>(request_indices);
    params.kv_tile_indices = const_cast<int32_t *>(kv_tile_indices);
    params.o_indptr = const_cast<int32_t *>(o_indptr);
    params.kv_chunk_size_ptr = const_cast<int32_t *>(kv_chunk_size);
    params.block_valid_mask = nullptr;

    cudaError_t st = cudaErrorInvalidValue;
    if (head_dim == 128) {
        st = launch_batch_decode_head_dim_dispatch<128, Params, Variant>(
            params, tmp_v, tmp_s, o_indptr_host, partition_kv, rowwise_merge, stream);
    } else if (head_dim == 256) {
        st = launch_batch_decode_head_dim_dispatch<256, Params, Variant>(
            params, tmp_v, tmp_s, o_indptr_host, partition_kv, rowwise_merge, stream);
    }
    if (st != cudaSuccess) return false;

    const uint64_t elems = static_cast<uint64_t>(batch) * n_heads * head_dim;
    const unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((elems + threads - 1) / threads);
    unpack_gate_batch_kernel<<<blocks, threads, 0, stream>>>(
        out, o_f16, q, q_stride, batch, n_heads, head_dim,
        q_batch_stride, out_batch_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool launch_batch_decode_f32q_f16kv(
        float *out,
        half *o_f16,
        half *tmp_v,
        float *tmp_s,
        const float *q,
        uint32_t q_stride,
        const void *k_cache,
        const void *v_cache,
        const int32_t *page_indices,
        const int32_t *page_indptr,
        const int32_t *last_page_len,
        const int32_t *request_indices,
        const int32_t *kv_tile_indices,
        const int32_t *o_indptr,
        const int32_t *o_indptr_host,
        const int32_t *kv_chunk_size,
        bool partition_kv,
        bool rowwise_merge,
        uint32_t page_size,
        uint32_t total_tiles,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t batch,
        uint32_t q_batch_stride,
        uint32_t out_batch_stride,
        float scale,
        cudaStream_t stream) {
    if (out == nullptr || o_f16 == nullptr || q == nullptr ||
        k_cache == nullptr || v_cache == nullptr || page_indices == nullptr ||
        page_indptr == nullptr || last_page_len == nullptr ||
        request_indices == nullptr || kv_tile_indices == nullptr ||
        o_indptr == nullptr || o_indptr_host == nullptr ||
        kv_chunk_size == nullptr) {
        return false;
    }
    if (partition_kv && (tmp_v == nullptr || tmp_s == nullptr)) {
        return false;
    }
    if (n_heads == 0 || n_kv_heads == 0 || (n_heads % n_kv_heads) != 0) return false;
    if (batch == 0 || total_tiles == 0 || page_size == 0 || q_stride < head_dim) return false;

    using Params = flashinfer::BatchDecodeParams<float, half, half, int32_t>;
    using Variant = flashinfer::DefaultAttention<false, false, false, false>;

    auto paged_kv = flashinfer::paged_kv_t<half, int32_t>(
        n_kv_heads, page_size, head_dim, batch, flashinfer::QKVLayout::kNHD,
        const_cast<half *>(static_cast<const half *>(k_cache)),
        const_cast<half *>(static_cast<const half *>(v_cache)),
        const_cast<int32_t *>(page_indices),
        const_cast<int32_t *>(page_indptr),
        const_cast<int32_t *>(last_page_len),
        nullptr);
    Params params(
        const_cast<float *>(q),
        nullptr,
        paged_kv,
        o_f16,
        nullptr,
        nullptr,
        n_heads,
        static_cast<int32_t>(q_batch_stride),
        static_cast<int32_t>(q_stride),
        -1,
        0.0f,
        scale,
        1.0f,
        1.0f);
    params.padded_batch_size = total_tiles;
    params.request_indices = const_cast<int32_t *>(request_indices);
    params.kv_tile_indices = const_cast<int32_t *>(kv_tile_indices);
    params.o_indptr = const_cast<int32_t *>(o_indptr);
    params.kv_chunk_size_ptr = const_cast<int32_t *>(kv_chunk_size);
    params.block_valid_mask = nullptr;

    cudaError_t st = cudaErrorInvalidValue;
    if (head_dim == 128) {
        st = launch_batch_decode_head_dim_dispatch<128, Params, Variant>(
            params, tmp_v, tmp_s, o_indptr_host, partition_kv, rowwise_merge, stream);
    } else if (head_dim == 256) {
        st = launch_batch_decode_head_dim_dispatch<256, Params, Variant>(
            params, tmp_v, tmp_s, o_indptr_host, partition_kv, rowwise_merge, stream);
    }
    if (st != cudaSuccess) return false;

    const uint64_t elems = static_cast<uint64_t>(batch) * n_heads * head_dim;
    const unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((elems + threads - 1) / threads);
    unpack_o_batch_kernel<<<blocks, threads, 0, stream>>>(
        out, o_f16, batch, n_heads, head_dim, out_batch_stride);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace flashinfer_adapter
} // namespace qw3
