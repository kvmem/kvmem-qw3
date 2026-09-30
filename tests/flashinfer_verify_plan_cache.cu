#include "qw3/device_backend.hpp"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qw3;
void check(DeviceStatus s) { if (!s.ok) throw std::runtime_error(s.message); }
void require(bool value, const char *why) { if (!value) throw std::runtime_error(why); }

int main(int argc, char **argv) try {
    const unsigned history = argc > 1 ? std::stoul(argv[1]) : 65536;
    require(history == 1024 || history == 65536 || history == 131072, "unfrozen history");
    constexpr unsigned dim = 256, heads = 24, kv_heads = 4, width = 8, page = 32;
    constexpr unsigned q_stride = heads * dim * 2, out_stride = heads * dim + 8;
    const unsigned pages = (history + width + 31) / page;
    auto device = make_cuda_device_backend(LinearBackend::Cublas);
    check(device->begin());
    std::vector<__nv_fp8_e4m3> k(uint64_t(pages) * page * kv_heads * dim), v(k.size());
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = __nv_fp8_e4m3(.5f * std::sin(float(i % 104729) * .13f));
        v[i] = __nv_fp8_e4m3(.5f * std::cos(float(i % 104723) * .17f));
    }
    std::vector<float> q(width * q_stride), sentinel(width * out_stride, 12345.f);
    for (size_t i = 0; i < q.size(); ++i) q[i] = std::sin(float(i) * .11f);
    std::vector<int32_t> mapping(pages);
    auto kd = device->tensor_fp8_kv(k.size(), "K");
    auto vd = device->tensor_fp8_kv(v.size(), "V");
    auto qd = device->tensor_f32(q.size(), "Q");
    auto pd = device->tensor_i32(pages, "mutable page table");
    auto out = device->tensor_f32(sentinel.size(), "guarded O");
    check(device->copy_bytes_from_host(*kd, 0, k.data(), k.size()));
    check(device->copy_bytes_from_host(*vd, 0, v.data(), v.size()));
    check(device->copy_bytes_from_host(*qd, 0, q.data(), q.size() * sizeof(float)));
    // The host-page API supports permuted pages only with FP16 KV. Give that
    // intervening call its own small synthetic buffer; the cache comparison
    // below continues to use FP8 KV and device page tables exclusively.
    constexpr unsigned scratch_pages = 9;
    std::vector<int32_t> scratch_mapping(scratch_pages);
    for (unsigned i = 0; i < scratch_pages; ++i) scratch_mapping[i] = scratch_pages - i - 1;
    std::vector<__half> scratch(scratch_pages * page * kv_heads * dim, __float2half(.25f));
    auto scratch_kv = device->tensor_f16(scratch.size(), "host-page metadata clobber KV");
    check(device->copy_bytes_from_host(*scratch_kv, 0, scratch.data(), scratch.size() * sizeof(__half)));
    auto clobber_metadata = [&] {
        const auto status = device->attention_decode_batch_paged_gated(*out, *qd, 2 * dim,
            *scratch_kv, *scratch_kv, scratch_mapping.data(), scratch_pages, page,
            heads, kv_heads, dim, 257, 3, q_stride, out_stride, 1.f / std::sqrt(float(dim)));
        if (!status.ok) throw std::runtime_error(std::string("host-page metadata clobber: ") + status.message);
    };
    auto call = [&](unsigned base, unsigned batch) {
        check(device->attention_decode_batch_paged_gated_device(*out, *qd, 2 * dim,
            *kd, *vd, *pd, pages, page, heads, kv_heads, dim,
            base, batch, q_stride, out_stride, 1.f / std::sqrt(float(dim))));
    };
    // Same shape/different page contents; an intervening API overwrites the
    // shared metadata buffer; widths and positions then change and roll back.
    const std::vector<std::pair<unsigned, unsigned>> shapes{
        {history, 8}, {history, 8}, {history, 8}, {history, 7},
        {history, 8}, {history, 1}, {history, 8}, {history - 1, 8}, {history, 8}};
    std::vector<std::vector<float>> expected;
    for (unsigned mode = 0; mode < 2; ++mode) {
        setenv("QW3_FLASHINFER_VERIFY_PLAN_CACHE", mode ? "1" : "0", 1);
        for (unsigned step = 0; step < shapes.size(); ++step) {
            std::cout << "plan_cache_check cache=" << mode << " step=" << step
                      << " history=" << history << std::endl;
            for (unsigned i = 0; i < pages; ++i) mapping[i] = (pages - i - 1 + step * 17) % pages;
            check(device->copy_i32_from_host(*pd, 0, mapping.data(), pages));
            if (step == 2) clobber_metadata();
            check(device->copy_bytes_from_host(*out, 0, sentinel.data(), sentinel.size() * sizeof(float)));
            call(shapes[step].first, shapes[step].second);
            // Repeated layers exercise actual cache hits without a host copy.
            for (unsigned layer = 0; layer < 16; ++layer) call(shapes[step].first, shapes[step].second);
            std::vector<float> values(sentinel.size());
            check(device->copy_to_host(*out, values.data(), 0, values.size()));
            for (unsigned row = 0; row < width; ++row) for (unsigned col = 0; col < out_stride; ++col) {
                const float value = values[row * out_stride + col];
                require(std::isfinite(value), "nonfinite attention");
                if (row >= shapes[step].second || col >= heads * dim)
                    require(value == 12345.f, "output guard overwritten");
            }
            if (!mode) expected.push_back(values);
            else require(values == expected[step], "cache changed FP8 causal attention output");
        }
    }
    std::cout << "plan_cache_exact_transitions=9 repeated_layers=16 history=" << history << '\n';
    // One-process operator timing only. End-to-end performance has a separate
    // frozen DeepSWE replay; these warm-resident KV calls are not that result.
    for (unsigned trial = 0; trial < 3; ++trial) {
        for (unsigned order = 0; order < 2; ++order) {
            const bool enabled = (trial + order) % 2;
            setenv("QW3_FLASHINFER_VERIFY_PLAN_CACHE", enabled ? "1" : "0", 1);
            call(history, 8); check(device->synchronize());
            const auto start = std::chrono::steady_clock::now();
            for (unsigned iteration = 0; iteration < 20; ++iteration)
                for (unsigned layer = 0; layer < 16; ++layer) call(history, 8);
            check(device->synchronize());
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            std::cout << "operator_timing trial=" << trial << " cache=" << enabled
                      << " calls=320 total_ms=" << ms << '\n';
        }
    }
    check(device->end());
    return 0;
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
