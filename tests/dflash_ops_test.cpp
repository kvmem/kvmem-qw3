#include "qw3/device_backend.hpp"
#include "dflash_model.hpp"
#include "dflash_weights.hpp"
#include "dflash_executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
using qw3::DeviceBackend;
using qw3::DeviceTensor;

void check(qw3::DeviceStatus status) {
    if (!status.ok) throw std::runtime_error(status.message);
}
void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
std::vector<float> random_values(size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<float> values(count);
    for (auto &v : values) v = float(int(rng() % 2048) - 1024) / 2048;
    return values;
}
template<class T>
std::unique_ptr<DeviceTensor> upload(DeviceBackend &backend, const std::vector<T> &values) {
    auto tensor = std::is_same<T, int32_t>::value
        ? backend.tensor_i32(values.size(), "dflash-test-i32")
        : backend.tensor_f32(values.size(), "dflash-test-f32");
    check(backend.copy_bytes_from_host(*tensor, 0, values.data(), values.size() * sizeof(T)));
    return tensor;
}
template<class T>
std::vector<T> download(DeviceBackend &backend, const DeviceTensor &tensor, size_t count) {
    std::vector<T> result(count);
    check(backend.copy_bytes_to_host(tensor, result.data(), 0, count * sizeof(T)));
    return result;
}
float compare(const std::vector<float> &actual, const std::vector<float> &expected,
              float tolerance, const char *name) {
    require(actual.size() == expected.size(), "comparison length mismatch");
    float error = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), "non-finite GPU result");
        error = std::max(error, std::abs(actual[i] - expected[i]));
    }
    std::cout << name << " max_abs_error=" << error << '\n';
    require(error <= tolerance, "GPU operator differs from double-precision oracle");
    return error;
}

void test_conv(DeviceBackend &backend, uint32_t block, uint32_t hidden, uint32_t group) {
    const uint32_t rows = 3 * block - 1, taps = 2, groups = hidden / group;
    const auto x = random_values(size_t(rows) * hidden, 5);
    const auto dynamic = random_values(size_t(rows) * 2 * taps * groups, 6);
    const auto base = random_values(2 * taps * hidden, 7);
    auto xd = upload(backend, x), dd = upload(backend, dynamic);
    auto bd = backend.weight_f32(base.data(), base.size(), "conv-base");
    auto output = backend.tensor_f32(x.size(), "conv-out");
    for (uint32_t side = 0; side < 2; ++side) {
        std::vector<float> reference(x.size());
        for (uint32_t r = 0; r < rows; ++r) for (uint32_t c = 0; c < hidden; ++c) {
            double value = 0;
            for (uint32_t tap = 0; tap < taps; ++tap) {
                if (tap > r % block) continue;
                const double coefficient = double(base[(side * taps + tap) * hidden + c]) +
                    dynamic[((r * 2 + side) * taps + tap) * groups + c / group];
                value += coefficient * x[(r - tap) * hidden + c];
            }
            reference[r * hidden + c] = value;
        }
        check(backend.dflash_grouped_conv(*output, *xd, *dd, *bd, rows, hidden, block, taps, group, side));
        compare(download<float>(backend, *output, x.size()), reference, 2e-6f, "conv");
    }
    require(!backend.dflash_grouped_conv(*xd, *xd, *dd, *bd, rows, hidden, block, taps, group, 0).ok,
            "unsafe in-place convolution accepted");
}

void test_attention(DeviceBackend &backend, uint32_t queries, uint32_t keys,
                    uint32_t heads, uint32_t kv_heads, uint32_t dim, uint32_t window) {
    auto q = random_values(size_t(queries) * heads * dim, 11);
    const auto k = random_values(size_t(keys) * kv_heads * dim, 12);
    const auto v = random_values(k.size(), 13);
    for (auto &x : q) x *= 4;
    auto qd = upload(backend, q), kd = upload(backend, k), vd = upload(backend, v);
    auto out = backend.tensor_f32(q.size(), "attention-out");
    std::vector<float> reference(q.size());
    for (uint32_t row = 0; row < queries; ++row) for (uint32_t head = 0; head < heads; ++head) {
        const int position = keys - queries + row;
        std::vector<double> scores(keys, 0);
        double maximum = -INFINITY;
        for (uint32_t t = 0; t < keys; ++t) {
            if (std::abs(int(t) - position) >= int(window)) { scores[t] = -INFINITY; continue; }
            double dot = 0;
            for (uint32_t d = 0; d < dim; ++d)
                dot += double(q[(row * heads + head) * dim + d]) *
                    k[(t * kv_heads + head / (heads / kv_heads)) * dim + d];
            scores[t] = dot / std::sqrt(double(dim));
            maximum = std::max(maximum, scores[t]);
        }
        double total = 0;
        for (auto &s : scores) { s = std::exp(s - maximum); total += s; }
        for (uint32_t d = 0; d < dim; ++d) {
            double value = 0;
            for (uint32_t t = 0; t < keys; ++t)
                value += scores[t] * v[(t * kv_heads + head / (heads / kv_heads)) * dim + d];
            reference[(row * heads + head) * dim + d] = value / total;
        }
    }
    check(backend.dflash_attention(*out, *qd, *kd, *vd, queries, keys, heads, kv_heads, dim, window));
    compare(download<float>(backend, *out, q.size()), reference, 3e-6f, "sliding-noncausal-attention");
}

void test_attention_batch(DeviceBackend &backend, uint32_t blocks, uint32_t queries) {
    constexpr uint32_t heads = 32, kv_heads = 8, dim = 128, window = 2048;
    const uint64_t width = uint64_t(queries) * heads * dim;
    auto query = random_values(blocks * width, 942 + blocks + queries);
    auto q = upload(backend, query);
    std::vector<std::unique_ptr<DeviceTensor>> owned_keys, owned_values;
    std::vector<const DeviceTensor *> keys, values;
    std::vector<uint32_t> lengths;
    std::vector<float> expected;
    for (uint32_t i = 0; i < blocks; ++i) {
        const uint32_t count = i % 4 == 0 ? queries : i % 4 == 1 ? queries + 5 : i % 4 == 2 ? 2048 + queries : 8192;
        owned_keys.push_back(upload(backend, random_values(uint64_t(count) * kv_heads * dim, 72 + i)));
        owned_values.push_back(upload(backend, random_values(uint64_t(count) * kv_heads * dim, 173 + i)));
        keys.push_back(owned_keys.back().get()); values.push_back(owned_values.back().get()); lengths.push_back(count);
        auto one_q = backend.tensor_view(*q, i * width * 4, width, 4, qw3::DeviceTensorDType::F32, "attention-private-q");
        require(bool(one_q), "invalid private attention view");
        auto one = backend.tensor_f32(width, "attention-private-result");
        check(backend.dflash_attention(*one, *one_q, *keys.back(), *values.back(), queries, count, heads, kv_heads, dim, window));
        const auto result = download<float>(backend, *one, width);
        expected.insert(expected.end(), result.begin(), result.end());
    }
    constexpr float guard = -4096;
    expected.insert(expected.end(), 13, guard);
    auto out = upload(backend, std::vector<float>(expected.size(), guard));
    auto metadata = backend.tensor_i32(uint64_t(blocks) * 6, "attention-request-metadata");
    for (bool reverse : {false, true}) {
        if (reverse) {
            std::reverse(keys.begin(), keys.end()); std::reverse(values.begin(), values.end()); std::reverse(lengths.begin(), lengths.end());
            auto reversed_q = query;
            auto reversed_out = expected;
            for (uint32_t i = 0; i < blocks; ++i) {
                std::copy_n(query.begin() + (blocks - 1 - i) * width, width, reversed_q.begin() + i * width);
                std::copy_n(expected.begin() + (blocks - 1 - i) * width, width, reversed_out.begin() + i * width);
            }
            check(backend.copy_bytes_from_host(*q, 0, reversed_q.data(), reversed_q.size() * 4));
            expected = std::move(reversed_out);
        }
        check(backend.dflash_attention_batch(*out, *q, *metadata, keys.data(), values.data(), lengths.data(),
            blocks, queries, heads, kv_heads, dim, window));
        compare(download<float>(backend, *out, expected.size()), expected, 0, "independent-cache-batched-attention");
    }
    lengths[0] = queries - 1;
    require(!backend.dflash_attention_batch(*out, *q, *metadata, keys.data(), values.data(), lengths.data(),
        blocks, queries, heads, kv_heads, dim, window).ok, "invalid request key count accepted");
    std::cout << "batched_attention blocks=" << blocks << " queries=" << queries << " reversed=1 bit_exact=1 guards=ok\n";
}

void test_topk(DeviceBackend &backend, uint32_t rows, uint32_t vocab, uint32_t top_k) {
    const auto logits = random_values(size_t(rows) * vocab, 23); // Includes exact ties.
    auto input = upload(backend, logits);
    const uint32_t parts = (vocab + 4095) / 4096;
    auto ids = backend.tensor_i32(size_t(rows) * top_k, "topk-ids");
    auto values = backend.tensor_f32(size_t(rows) * top_k, "topk-values");
    auto si = backend.tensor_i32(size_t(rows) * parts * top_k, "topk-scratch-ids");
    auto sv = backend.tensor_f32(size_t(rows) * parts * top_k, "topk-scratch-values");
    check(backend.dflash_topk(*ids, *values, *input, *si, *sv, rows, vocab, top_k));
    const auto actual_ids = download<int32_t>(backend, *ids, rows * top_k);
    const auto actual_values = download<float>(backend, *values, rows * top_k);
    for (uint32_t row = 0; row < rows; ++row) {
        std::vector<int32_t> order(vocab);
        std::iota(order.begin(), order.end(), 0);
        const float *score = logits.data() + size_t(row) * vocab;
        std::partial_sort(order.begin(), order.begin() + top_k, order.end(), [&](int a, int b) {
            return score[a] > score[b] || (score[a] == score[b] && a < b);
        });
        for (uint32_t i = 0; i < top_k; ++i) {
            require(actual_ids[row * top_k + i] == order[i], "top-k ID or tie-order mismatch");
            require(actual_values[row * top_k + i] == score[order[i]], "top-k score mismatch");
        }
    }
    std::cout << "topk rows=" << rows << " vocab=" << vocab << " passed\n";
}

std::vector<uint16_t> quantize_bf16(std::vector<float> &values) {
    std::vector<uint16_t> bits(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        uint32_t raw;
        std::memcpy(&raw, &values[i], 4);
        raw += 0x7fff + ((raw >> 16) & 1);
        bits[i] = raw >> 16;
        raw = uint32_t(bits[i]) << 16;
        std::memcpy(&values[i], &raw, 4);
    }
    return bits;
}

void test_capture(DeviceBackend &backend, bool bf16) {
    const uint32_t rows = 11, width = 513, stride = 520, taps = 5;
    auto values = random_values(rows * stride, 91);
    const auto bits = quantize_bf16(values);
    auto source = bf16 ? backend.tensor_bf16(values.size(), "capture-bf16") : upload(backend, values);
    if (bf16) check(backend.copy_bytes_from_host(*source, 0, bits.data(), bits.size() * 2));
    auto output = backend.tensor_f32(uint64_t(rows) * width * taps, "capture-features");
    std::vector<float> reference(output->count);
    for (uint32_t tap : {3u, 0u, 4u, 1u, 2u}) {
        check(backend.dflash_capture_features(*output, *source, rows, width, stride, taps, tap));
        for (uint32_t r = 0; r < rows; ++r) for (uint32_t col = 0; col < width; ++col)
            reference[(r * taps + tap) * width + col] = values[r * stride + col];
    }
    compare(download<float>(backend, *output, output->count), reference, 0,
            bf16 ? "feature-capture-bf16" : "feature-capture-fp32");
    require(!backend.dflash_capture_features(*output, *source, rows, width, stride, taps, taps).ok,
            "invalid target tap accepted");
}

void test_selector(DeviceBackend &backend, bool bf16) {
    const uint32_t blocks = 3, positions = 7, rows = blocks * positions;
    const uint32_t vocab = 73, rank = 256, top_k = 16;
    auto previous = random_values(vocab * rank, 41), next = random_values(vocab * rank, 42);
    const auto pb = quantize_bf16(previous), nb = quantize_bf16(next);
    auto pw = bf16 ? backend.weight_bf16(pb.data(), vocab, rank, "previous-codebook")
                   : backend.weight_f32(previous.data(), previous.size(), "previous-codebook");
    auto nw = bf16 ? backend.weight_bf16(nb.data(), vocab, rank, "next-codebook")
                   : backend.weight_f32(next.data(), next.size(), "next-codebook");
    pw->rows = nw->rows = vocab; pw->cols = nw->cols = rank;
    const auto hidden = random_values(rows * rank, 43), unary = random_values(rows * top_k, 44);
    std::vector<int32_t> candidates(rows * top_k), anchors = {0, 35, 72};
    std::vector<float> uniform(rows);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t i = 0; i < top_k; ++i) candidates[r * top_k + i] = (i * 7 + r * 13) % vocab;
        uniform[r] = float(r) / rows;
    }
    uniform.back() = std::nextafter(1.0f, 0.0f);
    auto cd = upload(backend, candidates), ad = upload(backend, anchors), ud = upload(backend, uniform);
    auto hd = upload(backend, hidden), un = upload(backend, unary);
    auto td = backend.tensor_i32(rows, "selected-tokens");
    auto pd = backend.tensor_f32(rows * top_k, "proposal-probabilities");
    for (const float temperature : {0.0f, 0.7f, 1.6f}) {
        std::vector<float> reference(rows * top_k, 0);
        std::vector<int32_t> tokens(rows);
        for (uint32_t block = 0; block < blocks; ++block) {
            int predecessor = anchors[block];
            for (uint32_t position = 0; position < positions; ++position) {
                const uint32_t row = block * positions + position;
                std::vector<double> score(top_k);
                for (uint32_t i = 0; i < top_k; ++i) {
                    score[i] = unary[row * top_k + i];
                    for (uint32_t d = 0; d < rank; ++d)
                        score[i] += double(previous[predecessor * rank + d]) * hidden[row * rank + d] *
                            next[candidates[row * top_k + i] * rank + d];
                }
                uint32_t chosen = std::max_element(score.begin(), score.end()) - score.begin();
                if (temperature > 0) {
                    const double maximum = score[chosen];
                    double sum = 0;
                    for (auto &s : score) { s = std::exp((s - maximum) / temperature); sum += s; }
                    double cdf = 0;
                    chosen = top_k - 1;
                    bool found = false;
                    for (uint32_t i = 0; i < top_k; ++i) {
                        reference[row * top_k + i] = score[i] / sum;
                        cdf += score[i] / sum;
                        if (!found && uniform[row] < cdf) { chosen = i; found = true; }
                    }
                } else reference[row * top_k + chosen] = 1;
                predecessor = candidates[row * top_k + chosen];
                tokens[row] = predecessor;
            }
        }
        check(backend.dflash_select(*td, *pd, *cd, *un, *hd, *pw, *nw, *ad, *ud,
            blocks, positions, top_k, rank, temperature));
        require(download<int32_t>(backend, *td, rows) == tokens, "conditional selector path mismatch");
        compare(download<float>(backend, *pd, rows * top_k), reference, 2e-6f,
                bf16 ? "selector-bf16" : "selector-fp32");
    }
}

void test_f32_linear(DeviceBackend &backend) {
    const uint32_t in = 17, out = 23, rows = 7;
    const auto w = random_values(in * out, 61), x = random_values(rows * in, 62);
    auto weight = backend.weight_f32(w.data(), w.size(), "reference-weight");
    weight->rows = out; weight->cols = in;
    auto input = upload(backend, x), result = backend.tensor_f32(rows * out, "reference-linear");
    std::vector<float> reference(rows * out);
    for (uint32_t r = 0; r < rows; ++r) for (uint32_t o = 0; o < out; ++o) {
        double sum = 0;
        for (uint32_t i = 0; i < in; ++i) sum += double(x[r * in + i]) * w[o * in + i];
        reference[r * out + o] = sum;
    }
    check(backend.linear_matmul(*result, *weight, *input, rows, in, out));
    compare(download<float>(backend, *result, rows * out), reference, 2e-6f, "fp32-linear");
}
} // namespace

int main(int argc, char **argv) try {
    auto backend = qw3::make_cuda_device_backend(qw3::LinearBackend::Cublas);
    check(backend->begin());
    test_conv(*backend, 8, 5120, 16);
    test_conv(*backend, 3, 28, 4);
    test_conv(*backend, 1, 16, 4);
    test_attention(*backend, 3, 5, 8, 2, 64, 2);
    test_attention(*backend, 8, 8, 32, 8, 128, 2048);
    test_attention(*backend, 8, 2056, 32, 8, 128, 2048);
    for (uint32_t blocks : {1u, 4u, 16u})
        for (uint32_t queries : {1u, 3u, 8u}) test_attention_batch(*backend, blocks, queries);
    test_topk(*backend, 3, 17, 16);
    test_topk(*backend, 3, 4097, 16);
    test_topk(*backend, 7, 248320, 16);
    test_selector(*backend, false);
    test_selector(*backend, true);
    test_f32_linear(*backend);
    test_capture(*backend, false);
    test_capture(*backend, true);
    if (argc == 2) {
        const qw3::DFlashModel model(argv[1]);
        const qw3::DFlashWeights weights(model, *backend);
        std::cout << "actual-checkpoint GPU upload bytes=" << weights.uploaded_bytes() << '\n';
        require(weights.at("norm.weight").format == qw3::DeviceWeightFormat::F32, "norm dtype");
        require(weights.at("layers.0.attention_conv.base_kernel").format == qw3::DeviceWeightFormat::F32, "conv dtype");
        require(weights.at("fc.weight").format == qw3::DeviceWeightFormat::BF16, "projection dtype");
        const auto &c = model.config();
        qw3::DFlashExecutor executor(weights, *backend);
        auto features = upload(*backend, random_values(size_t(3) * c.hidden_size * c.target_layer_ids.size(), 71));
        auto noise = upload(*backend, random_values(size_t(c.block_size) * c.hidden_size, 72));
        executor.append_context(*features, 3, 0);
        const auto first = download<float>(*backend, executor.forward(*noise, c.block_size, 3),
                                          size_t(c.block_size) * c.hidden_size);
        const auto repeated = download<float>(*backend, executor.forward(*noise, c.block_size, 3), first.size());
        compare(repeated, first, 0, "actual-BF16-draft-repeat");
        executor.reset();
        executor.append_context(*features, 3, 0);
        compare(download<float>(*backend, executor.forward(*noise, c.block_size, 3), first.size()),
                first, 0, "actual-BF16-draft-reset");
        std::cout << "actual checkpoint forward used synthetic features; no target generation benchmark\n";
    } else require(argc == 1, "usage: qw3-dflash-ops [draft directory]");
    check(backend->synchronize());
    std::cout << "DFlash CUDA operators passed\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}
