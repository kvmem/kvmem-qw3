#include "qw3/device_backend.hpp"
#include "flashinfer_decode_adapter.hpp"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <vector>

void check(qw3::DeviceStatus status) {
    if (!status.ok) throw std::runtime_error(status.message);
}

template<class KV>
void compare(qw3::DeviceBackend &backend, unsigned heads, unsigned kv_heads, bool fp8,
             bool prefill = false, unsigned dim = 256, unsigned batch = 4,
             unsigned base = 1001, bool cpu_oracle = true, bool prefill_serial = false) {
    constexpr unsigned page = 32;
    const unsigned pages = (base + batch + page - 1) / page;
    const unsigned stride = kv_heads * dim, q_stride = heads * 2 * dim;
    const unsigned out_stride = heads * dim + 8;
    constexpr float sentinel = 12345.0f;
    std::vector<KV> k(pages * page * stride), v(k.size());
    std::vector<float> query(batch * q_stride), result(batch * out_stride, sentinel);
    std::vector<int32_t> mapping(pages);
    for (unsigned p = 0; p < pages; ++p) mapping[p] = pages - p - 1;
    for (unsigned t = 0; t < pages * page; ++t) {
        const size_t offset = (mapping[t / page] * page + t % page) * stride;
        for (unsigned c = 0; c < stride; ++c) {
            k[offset + c] = KV(0.5f * std::sin(float(t * 3 + c) * 0.13f));
            v[offset + c] = KV(0.5f * std::cos(float(t + c * 2) * 0.17f));
        }
    }
    for (unsigned i = 0; i < query.size(); ++i) query[i] = std::sin(float(i) * 0.11f);
    auto kd = fp8 ? backend.tensor_fp8_kv(k.size(), "K") : backend.tensor_f16(k.size(), "K");
    auto vd = fp8 ? backend.tensor_fp8_kv(v.size(), "V") : backend.tensor_f16(v.size(), "V");
    auto qd = backend.tensor_f32(query.size(), "Q+gate");
    auto out = backend.tensor_f32(result.size(), "attention");
    auto pd = backend.tensor_i32(pages, "permuted pages");
    check(backend.copy_bytes_from_host(*kd, 0, k.data(), k.size() * sizeof(KV)));
    check(backend.copy_bytes_from_host(*vd, 0, v.data(), v.size() * sizeof(KV)));
    check(backend.copy_bytes_from_host(*qd, 0, query.data(), query.size() * sizeof(float)));
    check(backend.copy_i32_from_host(*pd, 0, mapping.data(), pages));
    std::vector<float> expected(result.size(), sentinel);
    for (unsigned b = 0; cpu_oracle && b < batch; ++b) {
        const unsigned length = base + b + 1;
        for (unsigned h = 0; h < heads; ++h) {
            const unsigned kh = h / (heads / kv_heads);
            const size_t qi = b * q_stride + h * 2 * dim;
            std::vector<double> scores(length);
            for (unsigned t = 0; t < length; ++t) {
                const size_t offset = (mapping[t / page] * page + t % page) * stride + kh * dim;
                double dot = 0;
                for (unsigned c = 0; c < dim; ++c) dot += double(query[qi + c]) * float(k[offset + c]);
                scores[t] = dot / std::sqrt(double(dim));
            }
            const double maximum = *std::max_element(scores.begin(), scores.end());
            double sum = 0;
            for (auto &score : scores) { score = std::exp(score - maximum); sum += score; }
            for (unsigned c = 0; c < dim; ++c) {
                double value = 0;
                for (unsigned t = 0; t < length; ++t) {
                    const size_t offset = (mapping[t / page] * page + t % page) * stride + kh * dim;
                    value += scores[t] * float(v[offset + c]);
                }
                value = value / sum / (1.0 + std::exp(-query[qi + dim + c]));
                expected[b * out_stride + h * dim + c] = float(value);
            }
        }
    }
    // A speculative verifier must preserve the single-token decoder's
    // numerics, including when batch width exceeds the number of KV tiles.
    std::vector<float> serial(result.size(), sentinel);
    if (!prefill || prefill_serial) {
        auto single_q = backend.tensor_f32(q_stride, "serial Q");
        auto single_out = backend.tensor_f32(out_stride, "serial O");
        for (unsigned b = 0; b < batch; ++b) {
            check(backend.copy_bytes_from_host(*single_q, 0, query.data() + b * q_stride,
                                               q_stride * sizeof(float)));
            check(backend.copy_bytes_from_host(*single_out, 0, serial.data() + b * out_stride,
                                               out_stride * sizeof(float)));
            const auto status = prefill_serial
                ? backend.attention_prefill_batch_paged_gated_device(
                    *single_out, *single_q, 2 * dim, *kd, *vd, *pd, pages, page,
                    heads, kv_heads, dim, base + b, 1, q_stride, out_stride,
                    1.0f / std::sqrt(float(dim)))
                : backend.attention_decode_batch_paged_gated_device(
                    *single_out, *single_q, 2 * dim, *kd, *vd, *pd, pages, page,
                    heads, kv_heads, dim, base + b, 1, q_stride, out_stride,
                    1.0f / std::sqrt(float(dim)));
            check(status);
            check(backend.copy_to_host(*single_out, serial.data() + b * out_stride, 0, out_stride));
        }
    }
    float max_error = 0;
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        std::fill(result.begin(), result.end(), sentinel);
        check(backend.copy_bytes_from_host(*out, 0, result.data(), result.size() * sizeof(float)));
        const auto status = prefill
            ? backend.attention_prefill_batch_paged_gated_device(
                *out, *qd, 2 * dim, *kd, *vd, *pd, pages, page, heads, kv_heads, dim,
                base, batch, q_stride, out_stride, 1.0f / std::sqrt(float(dim)))
            : backend.attention_decode_batch_paged_gated_device(
                *out, *qd, 2 * dim, *kd, *vd, *pd, pages, page, heads, kv_heads, dim,
                base, batch, q_stride, out_stride, 1.0f / std::sqrt(float(dim)));
        check(status);
        check(backend.copy_to_host(*out, result.data(), 0, result.size()));
        for (size_t i = 0; i < result.size(); ++i) {
            if (!std::isfinite(result[i])) throw std::runtime_error("non-finite attention output");
            if (i % out_stride >= heads * dim && result[i] != sentinel)
                throw std::runtime_error("attention overwrote output row guard");
            if (cpu_oracle) max_error = std::max(max_error, std::abs(result[i] - expected[i]));
            if ((!prefill || prefill_serial) && result[i] != serial[i])
                throw std::runtime_error(std::string(prefill_serial ? "paged prefill" : "paged decode") +
                    " differs between batched and single-token calls: base=" +
                    std::to_string(base) + " batch=" + std::to_string(batch) + " index=" + std::to_string(i));
        }
    }
    std::printf("mode=%s dim=%u batch=%u base=%u GQA=%u heads=%u kv=%u dtype=%s repeats=3 oracle=%s max_abs_error=%g\n",
                prefill ? "prefill" : "decode", dim, batch, base,
                heads / kv_heads, heads, kv_heads, fp8 ? "FP8" : "FP16",
                cpu_oracle ? "cpu" : "serial", max_error);
    std::fflush(stdout);
    if (max_error > 0.001f) throw std::runtime_error("paged causal attention differs from CPU oracle");
}

// Do not synchronize between layers. A cached plan still uploads request
// metadata, and the pinned host ring must not wrap while those reads are live.
template<class KV>
void queued_prefill(qw3::DeviceBackend &backend, bool fp8,
                    unsigned base = 8193, unsigned calls = 96) {
    constexpr unsigned dim = 256, heads = 24, kv_heads = 4, batch = 33;
    constexpr unsigned page = 32;
    const unsigned pages = (base + batch + 15 + page - 1) / page;
    constexpr unsigned stride = kv_heads * dim, q_stride = heads * 2 * dim;
    constexpr unsigned out_stride = heads * dim;
    std::vector<KV> k(pages * page * stride), v(k.size());
    std::vector<double> prefix(pages * page + 1);
    for (unsigned t = 0; t < pages * page; ++t) {
        const KV value = KV(float(t % 64) / 8);
        prefix[t + 1] = prefix[t] + float(value);
        for (unsigned c = 0; c < stride; ++c) v[t * stride + c] = value;
    }
    std::vector<int32_t> mapping(pages);
    for (unsigned p = 0; p < pages; ++p) mapping[p] = p;
    std::vector<float> query(batch * q_stride, 0);
    auto kd = fp8 ? backend.tensor_fp8_kv(k.size(), "queued K") : backend.tensor_f16(k.size(), "queued K");
    auto vd = fp8 ? backend.tensor_fp8_kv(v.size(), "queued V") : backend.tensor_f16(v.size(), "queued V");
    auto qd = backend.tensor_f32(query.size(), "queued Q");
    auto pd = backend.tensor_i32(pages, "queued pages");
    std::vector<std::unique_ptr<qw3::DeviceTensor>> outputs;
    for (unsigned i = 0; i < calls; ++i)
        outputs.push_back(backend.tensor_f32(batch * out_stride, "queued O"));
    check(backend.copy_bytes_from_host(*kd, 0, k.data(), k.size() * sizeof(KV)));
    check(backend.copy_bytes_from_host(*vd, 0, v.data(), v.size() * sizeof(KV)));
    check(backend.copy_bytes_from_host(*qd, 0, query.data(), query.size() * sizeof(float)));
    check(backend.copy_i32_from_host(*pd, 0, mapping.data(), pages));
    auto launch = [&](unsigned i) {
        check(backend.attention_prefill_batch_paged_gated_device(
            *outputs[i], *qd, 2 * dim, *kd, *vd, *pd, pages, page, heads,
            kv_heads, dim, base + i % 16, batch, q_stride, out_stride,
            1.0f / std::sqrt(float(dim))));
    };
    launch(0); // Grow scratch before the asynchronous burst.
    check(backend.synchronize());
    for (unsigned i = 0; i < calls; ++i) launch(i);
    check(backend.synchronize());
    std::vector<float> result(batch * out_stride);
    float max_error = 0;
    for (unsigned i = 0; i < calls; ++i) {
        check(backend.copy_to_host(*outputs[i], result.data(), 0, result.size()));
        for (unsigned b = 0; b < batch; ++b) {
            const unsigned length = base + i % 16 + b + 1;
            const float expected = float(prefix[length] / length / 2);
            for (unsigned d = 0; d < out_stride; ++d) {
                const float actual = result[b * out_stride + d];
                if (!std::isfinite(actual)) throw std::runtime_error("non-finite queued prefill");
                max_error = std::max(max_error, std::abs(actual - expected));
            }
        }
    }
    std::printf("queued dtype=%s base=%u calls=%u max_abs_error=%g\n", fp8 ? "FP8" : "FP16", base, calls, max_error);
    if (max_error > 0.001f) throw std::runtime_error("queued prefill metadata lifetime regression");
}

int main(int argc, char **argv) try {
    const bool prefill = argc == 2 && std::strcmp(argv[1], "--prefill") == 0;
    const bool serial_parity = argc == 2 && std::strcmp(argv[1], "--serial-parity") == 0;
    const bool serial_small = argc == 2 && std::strcmp(argv[1], "--serial-parity-small") == 0;
    const bool serial_long = argc == 2 && std::strcmp(argv[1], "--serial-parity-long") == 0;
    const bool prefill_serial = argc == 2 && std::strcmp(argv[1], "--prefill-serial-parity") == 0;
    const bool queued = argc == 2 && std::strcmp(argv[1], "--queued") == 0;
    const bool queued_small = argc == 2 && std::strcmp(argv[1], "--queued-small") == 0;
    if (argc != 1 && !prefill && !queued && !queued_small && !serial_parity && !serial_small && !serial_long && !prefill_serial)
        throw std::runtime_error("usage: flashinfer_decode_gqa [--prefill|--queued|--queued-small|--serial-parity|--serial-parity-small|--serial-parity-long]");
    auto backend = qw3::make_cuda_device_backend(qw3::LinearBackend::Cublas);
    check(backend->begin());
    if (prefill_serial) {
        // Feasibility gate for using the existing tensor-core prefill kernel
        // for BOTH one-token target execution and speculative verification.
        // Never switch production dispatch based solely on a fast batch path.
        unsigned failures = 0, cases = 0;
        for (unsigned batch : {2u, 4u, 8u}) {
            for (unsigned base : {1024u, 65597u, 65609u, 131209u, 131221u}) {
                ++cases;
                std::printf("prefill_serial_case base=%u batch=%u\n", base, batch);
                std::fflush(stdout);
                try {
                    compare<__nv_fp8_e4m3>(*backend, 24, 4, true, true, 256, batch, base, false, true);
                } catch (const std::exception &error) {
                    ++failures;
                    std::printf("prefill_serial_failure %s\n", error.what());
                    std::fflush(stdout);
                }
            }
        }
        std::printf("prefill_serial_cases=%u failures=%u performance_claim=false\n", cases, failures);
        return failures ? 1 : 0;
    }
    if (serial_long) {
        unsigned failures = 0;
        // Freeze the search ranges around the actual long KVMem/full-KV
        // lengths. Locate a partition change with the production planner,
        // then compare a nearby control and an eight-row crossing batch.
        for (unsigned cache : {0u, 1u}) {
            setenv("QW3_FLASHINFER_VERIFY_PLAN_CACHE", cache ? "1" : "0", 1);
            auto chunk = [](unsigned length) {
                return qw3::flashinfer_adapter::batch_decode_f32q_fp8kv_chunk_size(24, 4, 256, 1, length);
            };
            for (unsigned start : {65536u, 131072u}) {
                unsigned edge = start + 16;
                while (edge < start + 4096 && chunk(edge) == chunk(edge - 1)) ++edge;
                if (edge == start + 4096 || !chunk(edge))
                    throw std::runtime_error("long partition boundary not found");
                for (unsigned base : {edge - 16, edge - 4}) {
                    std::printf("long_serial_case cache=%u base=%u first_chunk=%u last_chunk=%u\n",
                        cache, base, chunk(base + 1), chunk(base + 8));
                    std::fflush(stdout);
                    try {
                        compare<__nv_fp8_e4m3>(*backend, 24, 4, true, false, 256, 8, base, false);
                    } catch (const std::exception &error) {
                        ++failures;
                        std::printf("long_serial_failure %s\n", error.what());
                        std::fflush(stdout);
                    }
                }
            }
        }
        std::printf("long_serial_failures=%u\n", failures);
        return failures ? 1 : 0;
    }
    if (serial_parity || serial_small) {
        // Race instrumentation is expensive. Its bounded variant retains both
        // KV dtypes and the problematic rows > tile-count merge regime.
        const std::vector<unsigned> dims = serial_small ? std::vector<unsigned>{256} : std::vector<unsigned>{128, 256};
        const std::vector<unsigned> widths = serial_small ? std::vector<unsigned>{8} : std::vector<unsigned>{2, 4, 5, 6, 7, 8};
        const std::vector<unsigned> bases = serial_small ? std::vector<unsigned>{256, 1024} : std::vector<unsigned>{255, 256, 511, 1021, 1024, 1277, 2047};
        for (unsigned dim : dims) for (unsigned batch : widths)
            for (unsigned base : bases) {
                compare<__nv_fp8_e4m3>(*backend, 24, 4, true, false, dim, batch, base, false);
                compare<half>(*backend, 24, 4, false, false, dim, batch, base, false);
            }
        return 0;
    }
    if (queued || queued_small) {
        // Race instrumentation retains extensive shared-memory access history.
        // Keep its bounded variant above two complete pinned-ring wraps; the
        // ordinary CTest and memcheck retain the full 8K/96-launch stress case.
        queued_prefill<half>(*backend, false, queued_small ? 257 : 8193, queued_small ? 24 : 96);
        queued_prefill<__nv_fp8_e4m3>(*backend, true, queued_small ? 257 : 8193, queued_small ? 24 : 96);
        return 0;
    }
    if (prefill) {
        for (unsigned dim : {128u, 256u}) {
            for (auto shape : {std::pair<unsigned, unsigned>{31, 0}, {33, 997}, {129, 63}}) {
                compare<__nv_fp8_e4m3>(*backend, 24, 4, true, true, dim, shape.first, shape.second);
                compare<half>(*backend, 24, 4, false, true, dim, shape.first, shape.second);
            }
        }
        return 0;
    }
    for (auto shape : {std::pair<unsigned, unsigned>{8, 2}, {16, 4}, {24, 4}}) {
        compare<__nv_fp8_e4m3>(*backend, shape.first, shape.second, true);
        compare<half>(*backend, shape.first, shape.second, false);
    }
    return 0;
} catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
