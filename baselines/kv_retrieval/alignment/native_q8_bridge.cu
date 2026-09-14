// SPDX-License-Identifier: MIT
// Thin ABI around the frozen qw3 Q8_0 kernels. Does not implement InfLLM.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "cuda_helpers.cuh"

namespace qw3::ported {
bool launch_quantize_q8_1(const float *, void *, uint32_t, uint32_t, uint32_t, cudaStream_t);
bool launch_quantize_mmq_q8_1(const float *, void *, uint32_t, uint32_t, uint32_t, cudaStream_t);
bool launch_mmvq_q8_0(const uint8_t *, const void *, float *, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t);
bool launch_mmq_q8_0(const uint8_t *, const void *, float *, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t);
bool launch_mmq_q8_0_medium(const uint8_t *, const void *, float *, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t);
}

// mode: 0 = MMVQ/FP16 activation scales; 1 = MMQ/FP32 activation scales;
// 2 = medium MMQ/FP32 scales. The Python boundary validates the layouts.
extern "C" int aligned_q8_matmul(const void *weight, const void *x, void *scratch,
                                 void *out, uint32_t rows, uint32_t cols,
                                 uint32_t batch, int mode, void *stream_ptr) {
    using namespace qw3::ported;
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    bool ok = mode == 0
        ? launch_quantize_q8_1(static_cast<const float *>(x), scratch, batch, cols, cols, stream)
        : launch_quantize_mmq_q8_1(static_cast<const float *>(x), scratch, batch, cols, cols, stream);
    if (!ok) return -1;
    auto w = static_cast<const uint8_t *>(weight);
    auto y = static_cast<float *>(out);
    if (mode == 0) ok = launch_mmvq_q8_0(w, scratch, y, rows, cols, batch, rows, stream);
    else if (mode == 1) ok = launch_mmq_q8_0(w, scratch, y, rows, cols, batch, rows, stream);
    else if (mode == 2) ok = launch_mmq_q8_0_medium(w, scratch, y, rows, cols, batch, rows, stream);
    else return -2;
    return ok ? static_cast<int>(cudaPeekAtLastError()) : -3;
}

__global__ void gather_q8_rows(const uint8_t *weights, const int64_t *ids,
                               float *out, uint32_t cols, uint64_t count,
                               uint32_t rows) {
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int64_t row = ids[i / cols];
    uint32_t col = i % cols;
    if (row < 0 || row >= rows) { out[i] = __int_as_float(0x7fffffff); return; }
    const uint64_t blocks = cols / 32;
    const auto *base = weights + row * blocks * 34;
    out[i] = __half2float(qw3::cuda_helpers::q8_d_plane(base)[col / 32]) *
             qw3::cuda_helpers::q8_qs_plane(base, blocks)[col];
}

extern "C" int aligned_q8_embedding(const void *weight, const void *ids, void *out,
                                    uint32_t rows, uint32_t cols, uint32_t batch,
                                    void *stream_ptr) {
    uint64_t count = static_cast<uint64_t>(cols) * batch;
    gather_q8_rows<<<(count + 255) / 256, 256, 0, reinterpret_cast<cudaStream_t>(stream_ptr)>>>(
        static_cast<const uint8_t *>(weight), static_cast<const int64_t *>(ids),
        static_cast<float *>(out), cols, count, rows);
    return static_cast<int>(cudaPeekAtLastError());
}
