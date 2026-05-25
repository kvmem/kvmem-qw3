#include "qw3/device_backend.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace qw3 {
namespace {

static thread_local char g_err[256];

DeviceStatus cuda_status(cudaError_t err, const char *where) {
    if (err == cudaSuccess) return {};
    std::snprintf(g_err, sizeof(g_err), "%s: %s", where, cudaGetErrorString(err));
    return {false, g_err};
}

DeviceStatus launch_status(const char *where) {
    return cuda_status(cudaGetLastError(), where);
}

DeviceStatus cublas_status(cublasStatus_t st, const char *where) {
    if (st == CUBLAS_STATUS_SUCCESS) return {};
    std::snprintf(g_err, sizeof(g_err), "%s: cublas status %d", where, static_cast<int>(st));
    return {false, g_err};
}

__device__ float fp16_to_f32_device(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

struct CudaTensor final : DeviceTensor {
    float *ptr = nullptr;
    std::string label;
    CudaTensor(uint64_t n, const char *name) {
        count = n;
        label = name ? name : "tensor";
        cudaMalloc(&ptr, static_cast<size_t>(n) * sizeof(float));
        cudaMemset(ptr, 0, static_cast<size_t>(n) * sizeof(float));
    }
    ~CudaTensor() override {
        if (ptr) cudaFree(ptr);
    }
};

enum class WeightType {
    F32,
    Q8_0,
};

struct CudaWeight final : DeviceWeight {
    void *ptr = nullptr;
    float *q8_f32_cache = nullptr;
    uint64_t bytes = 0;
    WeightType type = WeightType::F32;
    std::string label;
    CudaWeight(const void *src, uint64_t nbytes, uint64_t r, uint64_t c, WeightType t, const char *name) {
        rows = r;
        cols = c;
        bytes = nbytes;
        type = t;
        label = name ? name : "weight";
        cudaMalloc(&ptr, static_cast<size_t>(bytes));
        cudaMemcpy(ptr, src, static_cast<size_t>(bytes), cudaMemcpyHostToDevice);
    }
    ~CudaWeight() override {
        if (q8_f32_cache) cudaFree(q8_f32_cache);
        if (ptr) cudaFree(ptr);
    }
};

CudaTensor &as_tensor(DeviceTensor &t) {
    return static_cast<CudaTensor &>(t);
}

const CudaTensor &as_tensor(const DeviceTensor &t) {
    return static_cast<const CudaTensor &>(t);
}

const CudaWeight &as_weight(const DeviceWeight &w) {
    return static_cast<const CudaWeight &>(w);
}

__global__ void add_kernel(float *out, const float *a, const float *b, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void mul_kernel(float *out, const float *a, const float *b, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void silu_kernel(float *out, const float *x, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] / (1.0f + expf(-x[i]));
}

__global__ void rms_norm_kernel(float *out, const float *x, const float *weight, uint64_t n, float eps) {
    __shared__ float scratch[256];
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    for (uint64_t i = tid; i < n; i += blockDim.x) sum += x[i] * x[i];
    scratch[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(scratch[0] / static_cast<float>(n) + eps);
    for (uint64_t i = tid; i < n; i += blockDim.x) out[i] = x[i] * scale * weight[i];
}

__global__ void q8_get_row_kernel(float *out, const uint8_t *weight, uint64_t row, uint64_t cols) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cols) return;
    const uint64_t block = i / 32;
    const uint64_t inb = i % 32;
    const uint8_t *p = weight + (row * (cols / 32) + block) * 34;
    const uint16_t dh = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
    const int8_t q = reinterpret_cast<const int8_t *>(p + 2)[inb];
    out[i] = fp16_to_f32_device(dh) * static_cast<float>(q);
}

__global__ void q8_matvec_kernel(float *out, const uint8_t *weight, const float *x, uint64_t rows, uint64_t cols) {
    const uint64_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float scratch[256];
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    const uint64_t blocks = cols / 32;
    const uint8_t *rowp = weight + row * blocks * 34;
    for (uint64_t b = tid; b < blocks; b += blockDim.x) {
        const uint8_t *p = rowp + b * 34;
        const uint16_t dh = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
        const float d = fp16_to_f32_device(dh);
        const int8_t *qs = reinterpret_cast<const int8_t *>(p + 2);
        const float *xb = x + b * 32;
        for (uint32_t i = 0; i < 32; ++i) sum += d * static_cast<float>(qs[i]) * xb[i];
    }
    scratch[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[row] = scratch[0];
}

__global__ void q8_dequant_f32_kernel(float *out, const uint8_t *weight, uint64_t rows, uint64_t cols) {
    const uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = rows * cols;
    if (i >= n) return;
    const uint64_t row = i / cols;
    const uint64_t col = i % cols;
    const uint64_t block = col / 32;
    const uint64_t inb = col % 32;
    const uint8_t *p = weight + (row * (cols / 32) + block) * 34;
    const uint16_t dh = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
    const int8_t q = reinterpret_cast<const int8_t *>(p + 2)[inb];
    out[row * cols + col] = fp16_to_f32_device(dh) * static_cast<float>(q);
}

DeviceStatus dequant_q8_to_f32(float **out_ptr, const CudaWeight &w, bool persistent_cache) {
    if (!out_ptr) return {false, "dequant_q8_to_f32 out_ptr is null"};
    const uint64_t n = w.rows * w.cols;
    float *target = nullptr;
    if (persistent_cache && w.q8_f32_cache) {
        *out_ptr = w.q8_f32_cache;
        return {};
    }
    if (auto st = cuda_status(cudaMalloc(&target, static_cast<size_t>(n) * sizeof(float)), "cuda q8_f32 alloc"); !st.ok) {
        return st;
    }
    q8_dequant_f32_kernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(
        target, static_cast<const uint8_t *>(w.ptr), w.rows, w.cols);
    if (auto st = launch_status("cuda q8_f32 dequant"); !st.ok) {
        cudaFree(target);
        return st;
    }
    if (persistent_cache) {
        const_cast<CudaWeight &>(w).q8_f32_cache = target;
    }
    *out_ptr = target;
    return {};
}

DeviceStatus ensure_q8_f32_cache(CudaWeight &w) {
    if (w.q8_f32_cache) return {};
    if (w.type != WeightType::Q8_0) return {false, "ensure_q8_f32_cache requires Q8_0 weight"};
    float *ptr = nullptr;
    return dequant_q8_to_f32(&ptr, w, true);
}

// Causal 1D conv (single token, kernel size K).
// Inputs:
//   proj[c]         : current frame, [conv_dim]
//   state[c, k]     : K-1 previous frames, [conv_dim, K-1] indexed (c*(K-1) + k)
//                     state[c, 0] is the OLDEST frame, state[c, K-2] is the most-recent.
//   conv_w[c, k]    : weights, [conv_dim, K] indexed (c*K + k); tap K-1 applies to current.
// Outputs:
//   out[c]          : silu(sum_k conv_w[c, k] * frame[c, k])  where frame[c, K-1]=proj[c]
//   state           : shifted left, last slot now holds proj[c]
__global__ void recurrent_conv_kernel(float *out,
                                      float *state,
                                      const float *proj,
                                      const float *conv_w,
                                      uint32_t conv_dim,
                                      uint32_t conv_k) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    const float *w = conv_w + c * conv_k;
    float *st = state + c * (conv_k - 1);
    float acc = 0.0f;
    // Older frames live in state[0..K-2], current frame is proj[c].
    for (uint32_t k = 0; k + 1 < conv_k; ++k) acc += w[k] * st[k];
    const float cur = proj[c];
    acc += w[conv_k - 1] * cur;
    out[c] = acc / (1.0f + expf(-acc));
    // Shift state window: drop oldest, append current at the end.
    for (uint32_t k = 0; k + 2 < conv_k; ++k) st[k] = st[k + 1];
    st[conv_k - 2] = cur;
}

__global__ void l2_norm_128_kernel(float *x, uint32_t blocks, uint32_t stride, float eps) {
    const uint32_t b = blockIdx.x;
    if (b >= blocks) return;
    __shared__ float scratch[128];
    const uint32_t tid = threadIdx.x;
    float *base = x + b * stride;
    scratch[tid] = base[tid] * base[tid];
    __syncthreads();
    for (uint32_t s = 64; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    const float scale = rsqrtf(scratch[0] + eps);
    base[tid] *= scale;
}

__global__ void deltanet_kernel(float *core,
                                float *state,
                                const float *conv,
                                const float *alpha,
                                const float *beta,
                                const float *ssm_a,
                                const float *dt_bias,
                                uint32_t num_k_heads,
                                uint32_t num_v_heads,
                                uint32_t head_k_dim,
                                uint32_t head_v_dim) {
    const uint32_t vh = blockIdx.x;
    const uint32_t j = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (vh >= num_v_heads || j >= head_v_dim || tid >= head_k_dim) return;

    __shared__ float scratch[128];
    float *row = state + (static_cast<uint64_t>(vh) * head_v_dim + j) * head_k_dim;
    const float *q = conv + (vh % num_k_heads) * head_k_dim;
    const float *k = conv + num_k_heads * head_k_dim + (vh % num_k_heads) * head_k_dim;
    const float *v = conv + 2 * num_k_heads * head_k_dim + vh * head_v_dim;
    const float g_raw = log1pf(expf(alpha[vh] + dt_bias[vh])) * ssm_a[vh];
    const float eg = expf(g_raw);
    const float b = 1.0f / (1.0f + expf(-beta[vh]));

    row[tid] *= eg;
    scratch[tid] = row[tid] * k[tid];
    __syncthreads();
    for (uint32_t s = 64; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    const float delta = (v[j] - scratch[0]) * b;
    row[tid] += delta * k[tid];
    scratch[tid] = row[tid] * q[tid];
    __syncthreads();
    for (uint32_t s = 64; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    if (tid == 0) core[vh * head_v_dim + j] = scratch[0] * rsqrtf(static_cast<float>(head_v_dim));
}

__global__ void recurrent_norm_gate_kernel(float *core, const float *gate, const float *norm_w, uint32_t num_v_heads, uint32_t head_v_dim, float eps) {
    const uint32_t vh = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (vh >= num_v_heads || tid >= head_v_dim) return;
    __shared__ float scratch[256];
    float *block = core + vh * head_v_dim;
    scratch[tid] = block[tid] * block[tid];
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    const float scale = rsqrtf(scratch[0] / static_cast<float>(head_v_dim) + eps);
    const float z = gate[vh * head_v_dim + tid];
    block[tid] = block[tid] * scale * norm_w[tid] * (z / (1.0f + expf(-z)));
}

__global__ void rmsnorm_per_head_kernel(float *x,
                                        const float *w,
                                        uint32_t n_units,
                                        uint32_t per_unit_stride,
                                        uint32_t head_dim,
                                        float eps) {
    const uint32_t unit = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (unit >= n_units) return;
    __shared__ float scratch[256];
    float *base = x + unit * per_unit_stride;
    float sum = 0.0f;
    for (uint32_t i = tid; i < head_dim; i += blockDim.x) sum += base[i] * base[i];
    scratch[tid] = sum;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    const float scale = rsqrtf(scratch[0] / static_cast<float>(head_dim) + eps);
    for (uint32_t i = tid; i < head_dim; i += blockDim.x) base[i] = base[i] * scale * w[i];
}

__global__ void rope_partial_kernel(float *x,
                                    uint32_t n_units,
                                    uint32_t per_unit_stride,
                                    uint32_t rope_dim,
                                    uint32_t pos,
                                    float theta) {
    const uint32_t unit = blockIdx.x;
    const uint32_t i = threadIdx.x;
    if (unit >= n_units) return;
    const uint32_t half = rope_dim / 2;
    if (i >= half) return;
    float *base = x + unit * per_unit_stride;
    const float inv_freq = __powf(theta, -2.0f * static_cast<float>(i) / static_cast<float>(rope_dim));
    const float angle = static_cast<float>(pos) * inv_freq;
    float c, s;
    __sincosf(angle, &s, &c);
    const float x0 = base[i];
    const float x1 = base[i + half];
    base[i]        = x0 * c - x1 * s;
    base[i + half] = x0 * s + x1 * c;
}

__global__ void kv_append_kernel(float *cache, const float *src, uint32_t pos, uint32_t per_pos_size) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= per_pos_size) return;
    cache[static_cast<uint64_t>(pos) * per_pos_size + i] = src[i];
}

__global__ void attention_decode_qk_kernel(float *scores,
                                           const float *q,
                                           uint32_t q_stride,
                                           const float *k_cache,
                                           uint32_t n_heads,
                                           uint32_t n_kv_heads,
                                           uint32_t head_dim,
                                           uint32_t seq_len,
                                           float scale) {
    // grid: [n_heads, seq_len] ; block: 256 threads, tree-reduce dot prod.
    const uint32_t head = blockIdx.x;
    const uint32_t t = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (head >= n_heads || t >= seq_len) return;
    const uint32_t kv_head = head / (n_heads / n_kv_heads);
    const float *q_attn = q + head * q_stride; // first head_dim floats
    const float *k_t = k_cache + static_cast<uint64_t>(t) * n_kv_heads * head_dim + kv_head * head_dim;
    __shared__ float scratch[256];
    float acc = 0.0f;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) acc += q_attn[d] * k_t[d];
    scratch[tid] = acc;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    if (tid == 0) scores[static_cast<uint64_t>(head) * seq_len + t] = scratch[0] * scale;
}

__global__ void attention_decode_softmax_kernel(float *scores,
                                                uint32_t n_heads,
                                                uint32_t seq_len) {
    const uint32_t head = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (head >= n_heads) return;
    float *row = scores + static_cast<uint64_t>(head) * seq_len;
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float scratch[256];

    float local_max = -INFINITY;
    for (uint32_t t = tid; t < seq_len; t += blockDim.x) {
        if (row[t] > local_max) local_max = row[t];
    }
    scratch[tid] = local_max;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (scratch[tid + s] > scratch[tid]) scratch[tid] = scratch[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) s_max = scratch[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t t = tid; t < seq_len; t += blockDim.x) {
        const float v = expf(row[t] - s_max);
        row[t] = v;
        local_sum += v;
    }
    scratch[tid] = local_sum;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    if (tid == 0) s_sum = scratch[0];
    __syncthreads();

    const float inv = 1.0f / s_sum;
    for (uint32_t t = tid; t < seq_len; t += blockDim.x) row[t] *= inv;
}

__global__ void attention_decode_av_kernel(float *out,
                                           const float *scores,
                                           const float *v_cache,
                                           uint32_t n_heads,
                                           uint32_t n_kv_heads,
                                           uint32_t head_dim,
                                           uint32_t seq_len) {
    const uint32_t head = blockIdx.x;
    const uint32_t d = threadIdx.x;
    if (head >= n_heads || d >= head_dim) return;
    const uint32_t kv_head = head / (n_heads / n_kv_heads);
    const float *row = scores + static_cast<uint64_t>(head) * seq_len;
    float acc = 0.0f;
    for (uint32_t t = 0; t < seq_len; ++t) {
        const float *vt = v_cache + static_cast<uint64_t>(t) * n_kv_heads * head_dim + kv_head * head_dim;
        acc += row[t] * vt[d];
    }
    out[static_cast<uint64_t>(head) * head_dim + d] = acc;
}

__global__ void apply_attn_gate_kernel(float *out,
                                       const float *q,
                                       uint32_t q_stride,
                                       uint32_t n_heads,
                                       uint32_t head_dim) {
    const uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = static_cast<uint64_t>(n_heads) * head_dim;
    if (i >= total) return;
    const uint32_t head = static_cast<uint32_t>(i / head_dim);
    const uint32_t d = static_cast<uint32_t>(i % head_dim);
    const float gate = q[head * q_stride + head_dim + d];
    out[i] *= 1.0f / (1.0f + expf(-gate));
}

__global__ void attention_norm_mid_kernel(float *mid,
                                          float *q,
                                          float *k,
                                          const float *v,
                                          const float *q_norm,
                                          const float *k_norm,
                                          uint32_t n_heads,
                                          uint32_t n_kv_heads,
                                          uint32_t head_dim,
                                          float eps) {
    const uint32_t head = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (head >= n_heads || tid >= head_dim) return;
    __shared__ float scratch[256];

    float *q_base = q + head * 2 * head_dim;
    scratch[tid] = q_base[tid] * q_base[tid];
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    q_base[tid] *= rsqrtf(scratch[0] / static_cast<float>(head_dim) + eps) * q_norm[tid];
    __syncthreads();

    if (head < n_kv_heads) {
        float *k_base = k + head * head_dim;
        scratch[tid] = k_base[tid] * k_base[tid];
        __syncthreads();
        for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) scratch[tid] += scratch[tid + s];
            __syncthreads();
        }
        k_base[tid] *= rsqrtf(scratch[0] / static_cast<float>(head_dim) + eps) * k_norm[tid];
    }

    const uint32_t kvh = head % n_kv_heads;
    const float gate = q_base[head_dim + tid];
    mid[head * head_dim + tid] = v[kvh * head_dim + tid] * (1.0f / (1.0f + expf(-gate)));
}

class CudaDeviceBackend final : public DeviceBackend {
public:
    explicit CudaDeviceBackend(LinearBackend linear_backend)
        : linear_backend_(linear_backend) {}

    ~CudaDeviceBackend() override {
        if (cublas_handle_) cublasDestroy(cublas_handle_);
    }

    const char *name() const override {
        return "cuda-device";
    }

    DeviceStatus begin() override {
        if (auto st = cuda_status(cudaSetDevice(0), "cuda begin"); !st.ok) return st;
        if (!cublas_handle_) {
            if (auto st = cublas_status(cublasCreate(&cublas_handle_), "cublasCreate"); !st.ok) return st;
        }
        return {};
    }

    DeviceStatus end() override {
        return synchronize();
    }

    DeviceStatus synchronize() override {
        return cuda_status(cudaDeviceSynchronize(), "cuda synchronize");
    }

    std::unique_ptr<DeviceTensor> tensor_f32(uint64_t count, const char *label) override {
        return std::make_unique<CudaTensor>(count, label);
    }

    std::unique_ptr<DeviceWeight> weight_f32(const float *data, uint64_t count, const char *label) override {
        return std::make_unique<CudaWeight>(data, static_cast<uint64_t>(count * sizeof(float)), 1, count, WeightType::F32, label);
    }

    std::unique_ptr<DeviceWeight> weight_q8_0(const void *data, uint64_t rows, uint64_t cols, const char *label) override {
        return std::make_unique<CudaWeight>(data, rows * (cols / 32) * 34, rows, cols, WeightType::Q8_0, label);
    }

    DeviceStatus q8_0_get_row(DeviceTensor &out, const DeviceWeight &weight, uint64_t row) override {
        const auto &w = as_weight(weight);
        auto &o = as_tensor(out);
        q8_get_row_kernel<<<static_cast<unsigned>((w.cols + 255) / 256), 256>>>(o.ptr, static_cast<const uint8_t *>(w.ptr), row, w.cols);
        return launch_status("cuda q8_0_get_row");
    }

    DeviceStatus q8_0_matvec(DeviceTensor &out, const DeviceWeight &weight, const DeviceTensor &x) override {
        auto &w = const_cast<CudaWeight &>(as_weight(weight));
        auto &o = as_tensor(out);
        const auto &input = as_tensor(x);
        const bool use_cublas = linear_backend_ == LinearBackend::Cublas ||
            (linear_backend_ == LinearBackend::Auto && w.rows >= 1024 && w.cols >= 1024);
        if (use_cublas) {
            // Only take the cuBLAS path when the F32 dequant fits in our
            // persistent per-weight cache. Otherwise the previous behaviour
            // (cudaMalloc + dequant + sgemv + cudaFree on every call) blew up
            // wall time on big tensors like output.weight (~5 GiB F32 form).
            // For those, fall through to the custom Q8 matvec which reads the
            // mmap-uploaded Q8_0 blocks directly.
            const uint64_t f32_bytes = w.rows * w.cols * sizeof(float);
            const bool can_cache = f32_bytes <= (256ull << 20);
            if (can_cache) {
                if (auto st = ensure_q8_f32_cache(w); st.ok) {
                    const float alpha = 1.0f;
                    const float beta = 0.0f;
                    if (auto sgemv_st = cublas_status(
                            cublasSgemv(cublas_handle_,
                                        CUBLAS_OP_T,
                                        static_cast<int>(w.cols),
                                        static_cast<int>(w.rows),
                                        &alpha,
                                        w.q8_f32_cache,
                                        static_cast<int>(w.cols),
                                        input.ptr,
                                        1,
                                        &beta,
                                        o.ptr,
                                        1),
                            "cublasSgemv q8_0_matvec");
                        sgemv_st.ok) {
                        return {};
                    }
                }
                // cache build or sgemv failed: drop the error state and
                // fall through to the custom kernel.
                (void) cudaGetLastError();
            }
        }
        q8_matvec_kernel<<<static_cast<unsigned>(w.rows), 256>>>(o.ptr, static_cast<const uint8_t *>(w.ptr), input.ptr, w.rows, w.cols);
        return launch_status("cuda q8_0_matvec");
    }

    DeviceStatus rms_norm(DeviceTensor &out, const DeviceTensor &x, const DeviceWeight &weight, float eps) override {
        auto &o = as_tensor(out);
        const auto &input = as_tensor(x);
        const auto &w = as_weight(weight);
        rms_norm_kernel<<<1, 256>>>(o.ptr, input.ptr, static_cast<const float *>(w.ptr), input.count, eps);
        return launch_status("cuda rms_norm");
    }

    DeviceStatus add(DeviceTensor &out, const DeviceTensor &a, const DeviceTensor &b) override {
        auto &o = as_tensor(out);
        const auto &aa = as_tensor(a);
        const auto &bb = as_tensor(b);
        add_kernel<<<static_cast<unsigned>((o.count + 255) / 256), 256>>>(o.ptr, aa.ptr, bb.ptr, o.count);
        return launch_status("cuda add");
    }

    DeviceStatus silu(DeviceTensor &out, const DeviceTensor &x) override {
        auto &o = as_tensor(out);
        const auto &input = as_tensor(x);
        silu_kernel<<<static_cast<unsigned>((o.count + 255) / 256), 256>>>(o.ptr, input.ptr, o.count);
        return launch_status("cuda silu");
    }

    DeviceStatus mul(DeviceTensor &out, const DeviceTensor &a, const DeviceTensor &b) override {
        auto &o = as_tensor(out);
        const auto &aa = as_tensor(a);
        const auto &bb = as_tensor(b);
        mul_kernel<<<static_cast<unsigned>((o.count + 255) / 256), 256>>>(o.ptr, aa.ptr, bb.ptr, o.count);
        return launch_status("cuda mul");
    }

    DeviceStatus recurrent_single_token(DeviceTensor &core,
                                        DeviceTensor &state,
                                        DeviceTensor &conv_state,
                                        const DeviceTensor &proj,
                                        const DeviceTensor &gate,
                                        const DeviceTensor &alpha,
                                        const DeviceTensor &beta,
                                        const DeviceWeight &conv,
                                        const DeviceWeight &ssm_a,
                                        const DeviceWeight &dt_bias,
                                        const DeviceWeight &ssm_norm,
                                        uint32_t num_k_heads,
                                        uint32_t num_v_heads,
                                        uint32_t head_k_dim,
                                        uint32_t head_v_dim,
                                        uint32_t conv_kernel_size,
                                        float eps) override {
        auto &c = as_tensor(core);
        auto &s = as_tensor(state);
        auto &cs = as_tensor(conv_state);
        const auto &p = as_tensor(proj);
        const auto &g = as_tensor(gate);
        const auto &a = as_tensor(alpha);
        const auto &b = as_tensor(beta);
        const auto &cw = as_weight(conv);
        const auto &aw = as_weight(ssm_a);
        const auto &dt = as_weight(dt_bias);
        const auto &nw = as_weight(ssm_norm);
        float *conv_buf = nullptr;
        const size_t conv_bytes = static_cast<size_t>(p.count) * sizeof(float);
        if (auto st = cuda_status(cudaMalloc(&conv_buf, conv_bytes), "cuda recurrent temp"); !st.ok) return st;
        recurrent_conv_kernel<<<static_cast<unsigned>((p.count + 255) / 256), 256>>>(
            conv_buf, cs.ptr, p.ptr, static_cast<const float *>(cw.ptr),
            static_cast<uint32_t>(p.count), conv_kernel_size);
        l2_norm_128_kernel<<<num_k_heads, 128>>>(conv_buf, num_k_heads, head_k_dim, eps);
        l2_norm_128_kernel<<<num_k_heads, 128>>>(conv_buf + num_k_heads * head_k_dim, num_k_heads, head_k_dim, eps);
        dim3 grid(num_v_heads, head_v_dim);
        deltanet_kernel<<<grid, 128>>>(c.ptr,
                                       s.ptr,
                                       conv_buf,
                                       a.ptr,
                                       b.ptr,
                                       static_cast<const float *>(aw.ptr),
                                       static_cast<const float *>(dt.ptr),
                                       num_k_heads,
                                       num_v_heads,
                                       head_k_dim,
                                       head_v_dim);
        recurrent_norm_gate_kernel<<<num_v_heads, head_v_dim>>>(c.ptr, g.ptr, static_cast<const float *>(nw.ptr), num_v_heads, head_v_dim, eps);
        DeviceStatus st = launch_status("cuda recurrent_single_token");
        cudaFree(conv_buf);
        return st;
    }

    DeviceStatus zero_tensor(DeviceTensor &x) override {
        auto &t = as_tensor(x);
        return cuda_status(cudaMemset(t.ptr, 0, static_cast<size_t>(t.count) * sizeof(float)),
                            "zero_tensor");
    }

    DeviceStatus attention_single_token(DeviceTensor &mid,
                                        DeviceTensor &q,
                                        DeviceTensor &k,
                                        const DeviceTensor &v,
                                        const DeviceWeight &q_norm,
                                        const DeviceWeight &k_norm,
                                        uint32_t n_heads,
                                        uint32_t n_kv_heads,
                                        uint32_t head_dim,
                                        float eps) override {
        auto &m = as_tensor(mid);
        auto &qq = as_tensor(q);
        auto &kk = as_tensor(k);
        const auto &vv = as_tensor(v);
        const auto &qn = as_weight(q_norm);
        const auto &kn = as_weight(k_norm);
        attention_norm_mid_kernel<<<n_heads, head_dim>>>(m.ptr,
                                                         qq.ptr,
                                                         kk.ptr,
                                                         vv.ptr,
                                                         static_cast<const float *>(qn.ptr),
                                                         static_cast<const float *>(kn.ptr),
                                                         n_heads,
                                                         n_kv_heads,
                                                         head_dim,
                                                         eps);
        return launch_status("cuda attention_single_token");
    }

    DeviceStatus rmsnorm_per_head(DeviceTensor &x,
                                   const DeviceWeight &weight,
                                   uint32_t n_units,
                                   uint32_t per_unit_stride,
                                   uint32_t head_dim,
                                   float eps) override {
        auto &t = as_tensor(x);
        const auto &w = as_weight(weight);
        rmsnorm_per_head_kernel<<<n_units, 256>>>(t.ptr,
                                                  static_cast<const float *>(w.ptr),
                                                  n_units, per_unit_stride, head_dim, eps);
        return launch_status("cuda rmsnorm_per_head");
    }

    DeviceStatus rope_partial(DeviceTensor &x,
                               uint32_t n_units,
                               uint32_t per_unit_stride,
                               uint32_t rope_dim,
                               uint32_t pos,
                               float theta) override {
        auto &t = as_tensor(x);
        const uint32_t half = rope_dim / 2;
        if (half == 0) return {};
        rope_partial_kernel<<<n_units, half>>>(t.ptr, n_units, per_unit_stride, rope_dim, pos, theta);
        return launch_status("cuda rope_partial");
    }

    DeviceStatus kv_append(DeviceTensor &cache,
                           const DeviceTensor &src,
                           uint32_t pos,
                           uint32_t per_pos_size) override {
        auto &c = as_tensor(cache);
        const auto &s = as_tensor(src);
        const unsigned threads = 256;
        const unsigned blocks = (per_pos_size + threads - 1) / threads;
        kv_append_kernel<<<blocks, threads>>>(c.ptr, s.ptr, pos, per_pos_size);
        return launch_status("cuda kv_append");
    }

    DeviceStatus attention_decode(DeviceTensor &out,
                                   DeviceTensor &scores_scratch,
                                   const DeviceTensor &q,
                                   uint32_t q_stride,
                                   const DeviceTensor &k_cache,
                                   const DeviceTensor &v_cache,
                                   uint32_t n_heads,
                                   uint32_t n_kv_heads,
                                   uint32_t head_dim,
                                   uint32_t seq_len,
                                   float scale) override {
        auto &o = as_tensor(out);
        auto &s = as_tensor(scores_scratch);
        const auto &qq = as_tensor(q);
        const auto &kc = as_tensor(k_cache);
        const auto &vc = as_tensor(v_cache);
        dim3 grid_qk(n_heads, seq_len);
        attention_decode_qk_kernel<<<grid_qk, 256>>>(s.ptr, qq.ptr, q_stride, kc.ptr,
                                                    n_heads, n_kv_heads, head_dim, seq_len, scale);
        if (auto st = launch_status("cuda attn qk"); !st.ok) return st;
        attention_decode_softmax_kernel<<<n_heads, 256>>>(s.ptr, n_heads, seq_len);
        if (auto st = launch_status("cuda attn softmax"); !st.ok) return st;
        attention_decode_av_kernel<<<n_heads, head_dim>>>(o.ptr, s.ptr, vc.ptr,
                                                         n_heads, n_kv_heads, head_dim, seq_len);
        return launch_status("cuda attn av");
    }

    DeviceStatus apply_attn_gate(DeviceTensor &out,
                                  const DeviceTensor &q,
                                  uint32_t q_stride,
                                  uint32_t n_heads,
                                  uint32_t head_dim) override {
        auto &o = as_tensor(out);
        const auto &qq = as_tensor(q);
        const uint64_t total = static_cast<uint64_t>(n_heads) * head_dim;
        const unsigned threads = 256;
        const unsigned blocks = static_cast<unsigned>((total + threads - 1) / threads);
        apply_attn_gate_kernel<<<blocks, threads>>>(o.ptr, qq.ptr, q_stride, n_heads, head_dim);
        return launch_status("cuda apply_attn_gate");
    }

    DeviceArgmax argmax(const DeviceTensor &x) override {
        const auto &t = as_tensor(x);
        std::vector<float> host(static_cast<size_t>(t.count));
        cudaMemcpy(host.data(), t.ptr, static_cast<size_t>(t.count) * sizeof(float), cudaMemcpyDeviceToHost);
        DeviceArgmax best;
        best.logit = -INFINITY;
        for (size_t i = 0; i < host.size(); ++i) {
            if (host[i] > best.logit) {
                best.logit = host[i];
                best.token = static_cast<int>(i);
            }
        }
        return best;
    }

    DeviceStatus copy_to_host(const DeviceTensor &x, float *host, uint64_t offset, uint64_t count) override {
        const auto &t = as_tensor(x);
        if (offset + count > t.count) return {false, "copy_to_host out of range"};
        return cuda_status(cudaMemcpy(host,
                                      t.ptr + offset,
                                      static_cast<size_t>(count) * sizeof(float),
                                      cudaMemcpyDeviceToHost),
                            "copy_to_host");
    }

private:
    LinearBackend linear_backend_ = LinearBackend::Auto;
    cublasHandle_t cublas_handle_ = nullptr;
};

} // namespace

bool cuda_device_backend_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::unique_ptr<DeviceBackend> make_cuda_device_backend(LinearBackend linear_backend) {
    return std::make_unique<CudaDeviceBackend>(linear_backend);
}

} // namespace qw3
