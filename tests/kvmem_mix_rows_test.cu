#include "qw3/device_backend.hpp"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace qw3;
static void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
static void check(DeviceStatus status) { require(status.ok, status.message); }

template <typename T>
static void run(DeviceBackend &backend, DeviceTensorDType dtype) {
    constexpr uint32_t width = 37, capacity = 13, rows = 4;
    const std::vector<int32_t> indices{7, 1, 12, 3};
    auto allocate = [&](uint64_t n) {
        switch (dtype) {
        case DeviceTensorDType::F16: return backend.tensor_f16(n, "mix-test");
        case DeviceTensorDType::BF16: return backend.tensor_bf16(n, "mix-test");
        case DeviceTensorDType::FP8_E4M3: return backend.tensor_fp8_kv(n, "mix-test");
        default: return backend.tensor_f32(n, "mix-test");
        }
    };
    auto k = allocate(capacity * width), v = allocate(capacity * width);
    auto old_k = allocate(rows * width), old_v = allocate(rows * width);
    auto map = backend.tensor_i32(rows, "mix-test-rows");
    require(k->dtype == dtype && k->elem_size == sizeof(T), "unexpected test dtype");
    std::vector<T> fresh_k(k->count), fresh_v(v->count), before_k(old_k->count), before_v(old_v->count);
    for (size_t i = 0; i < fresh_k.size(); ++i) {
        fresh_k[i] = T(float(int(i % 71) - 35) * .13f);
        fresh_v[i] = T(float(int(i % 43) - 21) * .19f);
    }
    for (size_t i = 0; i < before_k.size(); ++i) {
        before_k[i] = T(float(int(i % 29) - 14) * .23f);
        before_v[i] = T(float(int(i % 31) - 15) * .17f);
    }
    auto put = [&](DeviceTensor &d, const std::vector<T> &values) {
        check(backend.copy_bytes_from_host(d, 0, values.data(), values.size() * sizeof(T)));
    };
    auto get = [&](const DeviceTensor &d) {
        std::vector<T> values(d.count);
        check(backend.copy_bytes_to_host(d, values.data(), 0, values.size() * sizeof(T)));
        return values;
    };
    check(backend.copy_i32_from_host(*map, 0, indices.data(), rows));
    put(*old_k, before_k); put(*old_v, before_v);
    for (float alpha : {0.0f, .25f, .5f, 1.0f}) {
        put(*k, fresh_k); put(*v, fresh_v);
        check(backend.mix_kv_rows(*k, *v, *old_k, *old_v, *map, rows, width, alpha));
        auto expect_k = fresh_k, expect_v = fresh_v;
        for (uint32_t row = 0; row < rows; ++row) for (uint32_t col = 0; col < width; ++col) {
            const auto at = indices[row] * width + col, packed = row * width + col;
            auto mix = [&](T fresh, T old) {
                if (alpha == 0) return old;
                if (alpha == 1) return fresh;
                volatile float left = alpha * float(fresh), right = (1 - alpha) * float(old);
                return T(left + right);
            };
            expect_k[at] = mix(fresh_k[at], before_k[packed]);
            expect_v[at] = mix(fresh_v[at], before_v[packed]);
        }
        const auto got_k = get(*k), got_v = get(*v);
        require(std::memcmp(got_k.data(), expect_k.data(), got_k.size() * sizeof(T)) == 0, "K mix/reference mismatch");
        require(std::memcmp(got_v.data(), expect_v.data(), got_v.size() * sizeof(T)) == 0, "V mix/reference mismatch");
        const auto saved_k = get(*old_k), saved_v = get(*old_v);
        require(std::memcmp(saved_k.data(), before_k.data(), saved_k.size() * sizeof(T)) == 0, "old K changed");
        require(std::memcmp(saved_v.data(), before_v.data(), saved_v.size() * sizeof(T)) == 0, "old V changed");
    }
    require(!backend.mix_kv_rows(*k, *v, *old_k, *old_v, *map, rows, width, -1).ok, "negative alpha accepted");
    require(!backend.mix_kv_rows(*k, *v, *old_k, *old_v, *map, rows, width, std::numeric_limits<float>::quiet_NaN()).ok, "NaN accepted");
    require(!backend.mix_kv_rows(*k, *v, *old_k, *old_v, *map, rows + 1, width, .5f).ok, "out-of-range map accepted");
    std::cout << "mix rows passed dtype=" << int(dtype) << '\n';
}

int main() {
    try {
        auto backend = make_cuda_device_backend(LinearBackend::Auto);
        require(bool(backend), "CUDA backend unavailable"); check(backend->begin());
        run<float>(*backend, DeviceTensorDType::F32);
        run<__half>(*backend, DeviceTensorDType::F16);
        run<__nv_bfloat16>(*backend, DeviceTensorDType::BF16);
        run<__nv_fp8_e4m3>(*backend, DeviceTensorDType::FP8_E4M3);
        check(backend->end()); return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
