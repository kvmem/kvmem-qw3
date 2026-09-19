#include "flashinfer_prefill_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct Buffer {
    void *data = nullptr;
    bool pinned;
    explicit Buffer(size_t bytes, bool host = false) : pinned(host) {
        check_cuda(host ? cudaMallocHost(&data, bytes) : cudaMalloc(&data, bytes));
    }
    ~Buffer() { if (pinned) cudaFreeHost(data); else cudaFree(data); }
    template<class T> T *as() { return static_cast<T *>(data); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
};

template<class KV>
void compare(unsigned dim, bool fp8, size_t float_bytes) {
    constexpr unsigned heads = 24, kv_heads = 4, batch = 33, base = 997, page = 32;
    constexpr unsigned pages = (base + batch + page - 1) / page;
    constexpr size_t int_bytes = 65536;
    const unsigned stride = kv_heads * dim, q_stride = heads * dim * 2;
    const unsigned out_stride = heads * dim + 8;
    constexpr float sentinel = 12345;
    std::vector<KV> kv(pages * page * stride);
    std::vector<double> prefix(pages * page + 1);
    for (unsigned t = 0; t < pages * page; ++t) {
        const KV value(float(t % 97) / 32);
        prefix[t + 1] = prefix[t] + float(value);
        std::fill_n(kv.begin() + t * stride, stride, value);
    }
    std::vector<int32_t> mapping(pages);
    for (unsigned p = 0; p < pages; ++p) mapping[p] = p;
    Buffer k(kv.size() * sizeof(KV)), v(kv.size() * sizeof(KV));
    Buffer q(batch * q_stride * sizeof(float)), qp(batch * heads * dim * sizeof(half));
    Buffer op(batch * heads * dim * sizeof(half)), out(batch * out_stride * sizeof(float));
    Buffer iw(int_bytes), host(int_bytes, true), fw(std::max<size_t>(float_bytes, 4));
    Buffer indices(pages * sizeof(int32_t));
    check_cuda(cudaMemset(k.data, 0, kv.size() * sizeof(KV)));
    check_cuda(cudaMemset(q.data, 0, batch * q_stride * sizeof(float)));
    check_cuda(cudaMemcpy(v.data, kv.data(), kv.size() * sizeof(KV), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(indices.data, mapping.data(), pages * sizeof(int32_t), cudaMemcpyHostToDevice));
    float maximum = 0;
    for (unsigned repeat = 0; repeat < 4; ++repeat) {
        // Model shared-scratch reuse/recycled allocations deterministically.
        // An address-keyed cache must not trust these old device contents.
        check_cuda(cudaMemset(iw.data, 0, int_bytes));
        std::fill_n(host.as<char>(), int_bytes, 0);
        std::vector<float> actual(batch * out_stride, sentinel);
        check_cuda(cudaMemcpy(out.data, actual.data(), actual.size() * sizeof(float), cudaMemcpyHostToDevice));
        bool hit = false;
        const unsigned current_base = base - repeat; // Same pages, different last-page length.
        const auto launch = fp8
            ? qw3::flashinfer_adapter::launch_batch_prefill_paged_f16q_fp8kv_gated
            : qw3::flashinfer_adapter::launch_batch_prefill_paged_f16q_f16kv_gated;
        if (!launch(out.as<float>(), qp.as<half>(), op.as<half>(), iw.as<int32_t>(),
                    host.data, int_bytes, float_bytes ? fw.as<float>() : nullptr, float_bytes,
                    q.as<float>(), 2 * dim, k.data, v.data, indices.as<int32_t>(),
                    pages, page, heads, kv_heads, dim, current_base, batch,
                    q_stride, out_stride, 1 / std::sqrt(float(dim)), nullptr, &hit))
            throw std::runtime_error("prefill launch failed");
        check_cuda(cudaDeviceSynchronize());
        if (repeat && !hit) throw std::runtime_error("regression did not exercise a cache hit");
        check_cuda(cudaMemcpy(actual.data(), out.data, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (unsigned b = 0; b < batch; ++b) {
            const unsigned length = current_base + b + 1;
            const float expected = float(prefix[length] / length / 2);
            for (unsigned d = 0; d < out_stride; ++d) {
                const float value = actual[b * out_stride + d];
                if (!std::isfinite(value)) throw std::runtime_error("non-finite cached prefill");
                if (d >= heads * dim) {
                    if (value != sentinel) throw std::runtime_error("output row guard overwritten");
                } else maximum = std::max(maximum, std::abs(value - expected));
            }
        }
    }
    std::printf("cache-restore dtype=%s dim=%u float_bytes=%zu max_abs_error=%g\n",
                fp8 ? "FP8" : "FP16", dim, float_bytes, maximum);
    if (maximum > 0.001f) throw std::runtime_error("cached plan workspace restoration differs from oracle");
}

int main() try {
    setenv("QW3_FLASHINFER_PLAN_CACHE", "1", 1);
    for (unsigned dim : {128u, 256u}) {
        for (size_t bytes : {size_t(0), size_t(16), size_t(64) << 20}) {
            compare<half>(dim, false, bytes);
            compare<__nv_fp8_e4m3>(dim, true, bytes);
        }
    }
    return 0;
} catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
