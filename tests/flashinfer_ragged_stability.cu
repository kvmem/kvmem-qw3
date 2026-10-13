#include "qw3/device_backend.hpp"
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace qw3;
static void check(DeviceStatus s) { if (!s.ok) throw std::runtime_error(s.message); }
static void require(bool v, const char *why) { if (!v) throw std::runtime_error(why); }

// Correctness only: a request's causal attention must survive short verifier
// tails, batch reordering and unrelated requests with different row counts.
int main() try {
    constexpr unsigned dim = 256, heads = 24, kv_heads = 4, width = 8;
    constexpr unsigned qs = heads * dim * 2, os = heads * dim + 8;
    auto device = make_cuda_device_backend(LinearBackend::Cublas);
    check(device->begin());
    unsigned comparisons = 0;
    for (unsigned page : {16u, 32u}) for (unsigned history : {256u, 2048u, 131072u}) {
        const unsigned pages_per_request = (history + width + page - 1) / page;
        const size_t kv_size = size_t(4) * pages_per_request * page * kv_heads * dim;
        std::vector<__nv_fp8_e4m3> host(kv_size);
        auto k = device->tensor_fp8_kv(kv_size, "K");
        auto v = device->tensor_fp8_kv(kv_size, "V");
        for (size_t i = 0; i < kv_size; ++i) host[i] = __nv_fp8_e4m3(.5f * std::sin(float(i % 65521) * .13f));
        check(device->copy_bytes_from_host(*k, 0, host.data(), host.size()));
        for (size_t i = 0; i < kv_size; ++i) host[i] = __nv_fp8_e4m3(.5f * std::cos(float(i % 65519) * .17f));
        check(device->copy_bytes_from_host(*v, 0, host.data(), host.size()));
        host.clear(); host.shrink_to_fit();
        auto q = device->tensor_f32(32 * qs, "Q");
        auto out = device->tensor_f32(33 * os, "guarded output");
        auto pi = device->tensor_i32(4 * pages_per_request, "pages");
        auto pp = device->tensor_i32(5, "page indptr");
        auto qp = device->tensor_i32(5, "Q indptr");
        auto last = device->tensor_i32(4, "last page");
        const std::vector<unsigned> lengths{2, 8, 1, 5};
        auto call = [&](const std::vector<unsigned> &order) {
            std::vector<int32_t> pages, pptr{0}, qptr{0}, lasts;
            std::vector<float> queries;
            for (auto req : order) {
                // Vary KV length too, including partially occupied last pages.
                const unsigned end = history - req * 3 + lengths[req];
                const unsigned np = (end + page - 1) / page;
                for (unsigned p = 0; p < np; ++p)
                    pages.push_back(req * pages_per_request + p);
                pptr.push_back(pages.size());
                qptr.push_back(qptr.back() + lengths[req]);
                lasts.push_back((end - 1) % page + 1);
                for (unsigned row = 0; row < lengths[req]; ++row)
                    for (unsigned col = 0; col < qs; ++col)
                        queries.push_back(std::sin(float((req * 8 + row) * qs + col) * .11f));
            }
            std::vector<float> values(33 * os, 12345.f);
            check(device->copy_bytes_from_host(*q, 0, queries.data(), queries.size() * sizeof(float)));
            check(device->copy_bytes_from_host(*out, 0, values.data(), values.size() * sizeof(float)));
            check(device->copy_i32_from_host(*pi, 0, pages.data(), pages.size()));
            check(device->copy_i32_from_host(*pp, 0, pptr.data(), pptr.size()));
            check(device->copy_i32_from_host(*qp, 0, qptr.data(), qptr.size()));
            check(device->copy_i32_from_host(*last, 0, lasts.data(), lasts.size()));
            check(device->attention_prefill_batch_paged_ragged_gated_device(*out, *q, 2 * dim,
                *k, *v, *pi, *pp, *last, *qp, qptr.data(), pptr.data(), order.size(),
                qptr.back(), page, heads, kv_heads, dim, qs, os, 1.f / std::sqrt(float(dim))));
            check(device->copy_to_host(*out, values.data(), 0, values.size()));
            for (unsigned row = 0; row < 33; ++row) for (unsigned col = 0; col < os; ++col) {
                require(std::isfinite(values[row * os + col]), "nonfinite attention");
                if (row >= unsigned(qptr.back()) || col >= heads * dim)
                    require(values[row * os + col] == 12345.f, "output guard overwritten");
            }
            std::vector<std::vector<float>> result(4);
            for (unsigned j = 0; j < order.size(); ++j)
                result[order[j]] = {values.begin() + qptr[j] * os, values.begin() + qptr[j + 1] * os};
            return result;
        };
        for (bool stable : {false, true}) {
            setenv("QW3_EXPERIMENTAL_FP8_STABLE_RAGGED_PREFILL", stable ? "1" : "0", 1);
            std::vector<std::vector<float>> reference(4);
            for (unsigned i = 0; i < 4; ++i) reference[i] = call({i})[i];
            size_t changed = 0;
            for (const auto &order : {std::vector<unsigned>{0, 1, 2, 3}, std::vector<unsigned>{3, 2, 0, 1}}) {
                const auto actual = call(order);
                for (unsigned i = 0; i < 4; ++i) for (size_t j = 0; j < reference[i].size(); ++j)
                    changed += actual[i][j] != reference[i][j];
                ++comparisons;
            }
            std::cout << "ragged_stability page=" << page << " history=" << history
                      << " fixed=" << stable << " changed_elements=" << changed << std::endl;
            if (stable) require(changed == 0, "fixed ragged attention changed with batch composition");
        }
    }
    check(device->end());
    std::cout << "ragged_stability comparisons=" << comparisons << " guards=pass status=pass\n";
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
