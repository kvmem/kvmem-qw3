#include "target_sampling_cuda.hpp"
#include "dflash_cuda.hpp"
#include <cub/cub.cuh>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cfloat>
#include <stdexcept>

namespace qw3 {
namespace {
template<class T> struct Buffer {
    T *data = nullptr;
    size_t capacity = 0;
    ~Buffer() { if (data) cudaFree(data); }
    cudaError_t reserve(size_t count) {
        if (count <= capacity) return cudaSuccess;
        if (data) { auto s = cudaFree(data); if (s != cudaSuccess) return s; data = nullptr; capacity = 0; }
        auto s = cudaMalloc(reinterpret_cast<void **>(&data), count * sizeof(T));
        if (s == cudaSuccess) capacity = count;
        return s;
    }
};
struct Config {
    uint32_t rows, vocab, cap, rank_stride, candidates;
    float temperature, top_p, min_p, presence, repetition;
    bool top_k_first;
    bool residual_always_draw;
};
struct RowInfo { double total; float norm; int positive, fallback; };
struct Answer { int accepted, token, draws, fallback; };

__global__ void mark_seen(int *seen, const uint32_t *ids, uint32_t n, uint32_t vocab) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && ids[i] < vocab) atomicExch(seen + ids[i], 1);
}
__global__ void penalize(float *out, int *ids, const float *logits, const int *seen,
                          const int *drafts, int *invalid, Config c) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= uint64_t(c.rows) * c.vocab) return;
    const uint32_t row = i / c.vocab, token = i % c.vocab;
    bool present = seen[token];
    for (uint32_t j = 0; j < row; ++j) present |= drafts[j] == int(token);
    float value = logits[i];
    if (present) {
        if (c.repetition > 0 && c.repetition != 1)
            value = value >= 0 ? __fdiv_rn(value, c.repetition) : __fmul_rn(value, c.repetition);
        value = __fsub_rn(value, c.presence);
    }
    if (isnan(value) || value == CUDART_INF_F) { atomicOr(invalid + row, 4); value = -FLT_MAX; }
    out[i] = value; ids[i] = token;
}
__device__ float weight(float logit, float maximum, float temperature) {
    return expf(__fmul_rn(__fsub_rn(logit, maximum), __fdiv_rn(1.f, temperature)));
}
__global__ void normalizer(RowInfo *info, const float *logits, const float *ranked, const int *invalid, Config c) {
    using Reduce = cub::BlockReduce<double, 256>;
    __shared__ typename Reduce::TempStorage scratch;
    const uint32_t row = blockIdx.x;
    const float maximum = ranked[uint64_t(row) * c.rank_stride];
    const bool truncate = c.top_k_first && c.cap < c.vocab;
    const uint32_t count = truncate ? c.cap : c.vocab;
    const float *values = truncate ? ranked + uint64_t(row) * c.rank_stride : logits + uint64_t(row) * c.vocab;
    double sum = 0;
    if (c.temperature > 0)
        for (uint32_t i = threadIdx.x; i < count; i += 256) sum += weight(values[i], maximum, c.temperature);
    sum = Reduce(scratch).Sum(sum);
    if (!threadIdx.x) {
        info[row] = {};
        info[row].fallback = invalid[row];
        info[row].norm = float(1.0 / (sum > 0 ? sum : 1.0));
        if (!isfinite(maximum) || (c.temperature > 0 && !isfinite(sum))) info[row].fallback = 4;
    }
}
// Filter only the ranked candidate set: usually 20 entries, independent of V.
// Full-sort requests also use this exact contract; the common small-k path
// avoids sorting the vocabulary and all serving paths avoid a vocabulary D2H.
__global__ void filter(float *prob, RowInfo *info, const float *ranked, const int *ids, Config c) {
    if (threadIdx.x) return;
    const uint32_t row = blockIdx.x;
    const float *logits = ranked + uint64_t(row) * c.rank_stride;
    const int *tokens = ids + uint64_t(row) * c.rank_stride;
    float *p = prob + uint64_t(row) * c.vocab;
    auto &r = info[row];
    if (c.temperature <= 0 || r.fallback) {
        p[tokens[0]] = 1; r.total = 1; r.positive = 1; return;
    }
    auto probability = [&](uint32_t i) { return __fmul_rn(weight(logits[i], logits[0], c.temperature), r.norm); };
    uint32_t keep = c.cap;
    // CPU nth_element does not specify which equal-logit tokens cross top-k.
    // Retain that exact legacy behavior through an explicit CPU fallback.
    if (c.cap < c.vocab && isfinite(logits[c.cap - 1]) && logits[c.cap - 1] == logits[c.cap]) r.fallback |= 1;
    if (c.min_p > 0) {
        const float cutoff = __fmul_rn(probability(0), c.min_p);
        keep = 0;
        while (keep < c.cap && probability(keep) >= cutoff) ++keep;
        for (uint32_t i = keep ? keep - 1 : 0; i <= keep && i < c.cap; ++i)
            if (fabsf(probability(i) - cutoff) <= 2e-6f * probability(0)) r.fallback |= 2;
    }
    if (c.top_p > 0 && c.top_p < 1) {
        double cumulative = 0;
        uint32_t cut = keep;
        for (uint32_t i = 0; i < keep; ++i) {
            cumulative += probability(i);
            if (fabs(cumulative - c.top_p) <= 2e-6) r.fallback |= 2;
            if (cumulative >= c.top_p) { cut = i + 1; break; }
        }
        if (c.top_k_first && cut) {
            const float cutoff = probability(cut - 1);
            while (cut < keep && probability(cut) == cutoff) ++cut;
        } else if (cut && cut < keep && logits[cut - 1] == logits[cut]) r.fallback |= 1;
        keep = cut;
    }
    double sum = 0;
    for (uint32_t i = 0; i < keep; ++i) sum += probability(i);
    if (!(sum > 0)) {
        if (c.vocab > 1 && logits[0] == logits[1]) r.fallback |= 1;
        p[tokens[0]] = 1; r.total = 1; r.positive = 1; return;
    }
    const bool truncate = c.cap < c.vocab || (c.top_p > 0 && c.top_p < 1) || c.min_p > 0;
    const float norm = truncate ? float(1.0 / sum) : 1.f;
    r.total = 0; r.positive = 0;
    for (uint32_t i = 0; i < keep; ++i) {
        const float value = __fmul_rn(probability(i), norm);
        p[tokens[i]] = value; r.total += value; r.positive += value > 0;
    }
}
__global__ void full_softmax(float *prob, const float *logits, const float *maximum, const RowInfo *info, Config c) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= uint64_t(c.rows) * c.vocab) return;
    const uint32_t row = i / c.vocab;
    prob[i] = __fmul_rn(weight(logits[i], maximum[uint64_t(row) * c.rank_stride], c.temperature), info[row].norm);
}
__global__ void full_statistics(RowInfo *info, const float *prob, Config c) {
    using Reduce = cub::BlockReduce<double, 256>;
    __shared__ typename Reduce::TempStorage scratch;
    const uint32_t row = blockIdx.x;
    double total = 0, positive = 0;
    for (uint32_t i = threadIdx.x; i < c.vocab; i += 256) {
        const float p = prob[uint64_t(row) * c.vocab + i]; total += p; positive += p > 0;
    }
    total = Reduce(scratch).Sum(total);
    __syncthreads();
    positive = Reduce(scratch).Sum(positive);
    if (!threadIdx.x) { info[row].total = total; info[row].positive = int(positive); }
}
__global__ void decide(Answer *out, const float *prob, const RowInfo *info, const int *drafts,
                        const int *ids, const float *q, const double *uniforms, Config c) {
    if (threadIdx.x) return;
    Answer a{0, -1, 0, 0};
    for (uint32_t row = 0; row + 1 < c.rows; ++row) {
        a.fallback |= info[row].fallback;
        double total = 0, selected = 0;
        for (uint32_t j = 0; j < c.candidates; ++j) {
            const uint64_t i = uint64_t(row) * c.candidates + j;
            total += q[i]; if (ids[i] == drafts[row]) selected = q[i];
        }
        const double alpha = fmin(1.0, (prob[uint64_t(row) * c.vocab + drafts[row]] / info[row].total) / (selected / total));
        if (alpha > 0 && alpha < 1 && fabs(uniforms[row] - alpha) < 2e-6) a.fallback |= 8;
        ++a.draws;
        if (!(uniforms[row] < alpha)) {
            const int residual_positive = info[row].positive - (prob[uint64_t(row) * c.vocab + drafts[row]] > 0);
            a.draws += c.residual_always_draw || residual_positive > 1;
            *out = a; return;
        }
        ++a.accepted;
    }
    a.fallback |= info[c.rows - 1].fallback;
    a.draws += info[c.rows - 1].positive > 1;
    *out = a;
}
__global__ void residual(double *weights, double *sums, const float *prob, const RowInfo *info,
                          const Answer *answer, const int *ids, const float *q, Config c) {
    using Reduce = cub::BlockReduce<double, 256>;
    __shared__ typename Reduce::TempStorage scratch;
    const uint32_t row = answer->accepted, token = blockIdx.x * 256 + threadIdx.x;
    double value = 0;
    if (token < c.vocab) {
        value = prob[uint64_t(row) * c.vocab + token] / info[row].total;
        if (row + 1 < c.rows) {
            double total = 0, subtract = 0;
            for (uint32_t j = 0; j < c.candidates; ++j) {
                const uint64_t i = uint64_t(row) * c.candidates + j;
                total += q[i]; if (ids[i] == int(token)) subtract = q[i];
            }
            value = fmax(0.0, value - subtract / total);
        }
        weights[token] = value;
    }
    const double sum = Reduce(scratch).Sum(value);
    if (!threadIdx.x) sums[blockIdx.x] = sum;
}
__global__ void draw(Answer *out, const double *weights, const double *sums, const double *uniforms, Config c) {
    if (threadIdx.x) return;
    const uint32_t blocks = (c.vocab + 255) / 256;
    double total = 0;
    for (uint32_t b = 0; b < blocks; ++b) total += sums[b];
    if (!(total > 0)) { out->fallback |= 4; return; }
    const uint32_t index = out->accepted + (out->accepted + 1 < int(c.rows));
    const double target = uniforms[index] * total;
    double cumulative = 0;
    int last = -1;
    for (uint32_t b = 0; b < blocks; ++b) {
        if (cumulative + sums[b] > target || b + 1 == blocks) {
            for (uint32_t token = b * 256; token < min(c.vocab, (b + 1) * 256); ++token) {
                if (weights[token] <= 0) continue;
                if (fabs(target - cumulative) < 2e-8 * total) out->fallback |= 8;
                last = token; cumulative += weights[token];
                if (target < cumulative) {
                    // A boundary very close to the draw is checked on CPU.
                    if (fabs(target - cumulative) < 2e-8 * total) out->fallback |= 8;
                    out->token = token; return;
                }
            }
        } else cumulative += sums[b];
    }
    out->token = last;
}
} // namespace

struct TargetSamplerCuda::Storage {
    Buffer<float> adjusted, ranked, probs, top_values, q;
    Buffer<int> ids, ranked_ids, top_ids, seen, drafts, candidates, offsets, invalid;
    Buffer<uint32_t> seen_ids;
    Buffer<double> uniforms, weights, sums;
    Buffer<RowInfo> info;
    Buffer<Answer> answer;
    Buffer<char> sort;
};
TargetSamplerCuda::TargetSamplerCuda() : storage_(std::make_unique<Storage>()) {}
TargetSamplerCuda::~TargetSamplerCuda() = default;

cudaError_t TargetSamplerCuda::run(const float *logits, const DeviceSamplingRequest &r,
                                   DeviceSamplingResult &result, cudaStream_t stream) {
    const uint32_t rows = r.drafts.size() + 1;
    if (!r.vocab || rows > 64 || uint64_t(rows) * r.vocab > INT_MAX || r.uniforms.size() < rows + 1 ||
        !std::isfinite(r.temperature) || !std::isfinite(r.top_p) || !std::isfinite(r.min_p) ||
        !std::isfinite(r.presence_penalty) || !std::isfinite(r.repetition_penalty)) return cudaErrorInvalidValue;
    for (double u : r.uniforms) if (!(u >= 0 && u < 1)) return cudaErrorInvalidValue;
    if (rows > 1) {
        if (!r.candidates_per_row || r.candidates_per_row > 32 ||
            r.candidate_ids.size() != uint64_t(rows - 1) * r.candidates_per_row ||
            r.proposal_probabilities.size() != r.candidate_ids.size()) return cudaErrorInvalidValue;
        for (uint32_t i = 0; i + 1 < rows; ++i) {
            double total = 0, selected = 0;
            for (uint32_t j = 0; j < r.candidates_per_row; ++j) {
                const auto k = uint64_t(i) * r.candidates_per_row + j;
                const int id = r.candidate_ids[k]; const float p = r.proposal_probabilities[k];
                if (id < 0 || uint32_t(id) >= r.vocab || p < 0 || !std::isfinite(p)) return cudaErrorInvalidValue;
                for (uint32_t m = 0; m < j; ++m)
                    if (r.candidate_ids[uint64_t(i) * r.candidates_per_row + m] == id) return cudaErrorInvalidValue;
                total += p; if (id == r.drafts[i]) selected = p;
            }
            if (!(total > 0 && selected > 0)) return cudaErrorInvalidValue;
        }
    }
    const uint32_t cap = r.temperature <= 0 ? 1 : r.top_k > 0 ? std::min(uint32_t(r.top_k), r.vocab) : r.vocab;
    const uint32_t parts = (r.vocab + 4095) / 4096;
    const bool unfiltered = r.temperature > 0 && cap == r.vocab &&
        !(r.top_p > 0 && r.top_p < 1) && !(r.min_p > 0);
    const uint32_t small_k = unfiltered ? 1 : std::min(r.vocab, cap + uint32_t(cap < r.vocab));
    const bool fast = small_k <= 32 && uint64_t(parts) * small_k <= 4096;
    const uint32_t stride = fast ? small_k : r.vocab;
    const uint64_t elements = uint64_t(rows) * r.vocab;
    Config c{rows, r.vocab, cap, stride, r.candidates_per_row, r.temperature, r.top_p, r.min_p,
             r.presence_penalty, r.repetition_penalty, r.top_k_first, r.residual_always_draw};
    auto &s = *storage_;
    Answer answer;
    std::vector<int> offsets;
    std::vector<RowInfo> report_info;
    // Keep pageable upload/download storage alive even if a later launch fails.
    struct SyncOnExit {
        cudaStream_t stream;
        bool done = false;
        ~SyncOnExit() { if (!done) cudaStreamSynchronize(stream); }
    } sync{stream};
#define CHECK_CUDA(expr) do { auto status = (expr); if (status != cudaSuccess) return status; } while (0)
    CHECK_CUDA(s.adjusted.reserve(elements)); CHECK_CUDA(s.ids.reserve(elements)); CHECK_CUDA(s.probs.reserve(elements));
    CHECK_CUDA(s.ranked.reserve(uint64_t(rows) * stride)); CHECK_CUDA(s.ranked_ids.reserve(uint64_t(rows) * stride));
    CHECK_CUDA(s.seen.reserve(r.vocab)); CHECK_CUDA(s.seen_ids.reserve(r.seen.size()));
    CHECK_CUDA(s.drafts.reserve(rows)); CHECK_CUDA(s.candidates.reserve(r.candidate_ids.size()));
    CHECK_CUDA(s.q.reserve(r.proposal_probabilities.size())); CHECK_CUDA(s.uniforms.reserve(r.uniforms.size()));
    CHECK_CUDA(s.info.reserve(rows)); CHECK_CUDA(s.answer.reserve(1));
    CHECK_CUDA(s.invalid.reserve(rows)); CHECK_CUDA(cudaMemsetAsync(s.invalid.data, 0, rows * sizeof(int), stream));
    CHECK_CUDA(s.weights.reserve(r.vocab)); CHECK_CUDA(s.sums.reserve((r.vocab + 255) / 256));
    CHECK_CUDA(cudaMemsetAsync(s.seen.data, 0, r.vocab * sizeof(int), stream));
    CHECK_CUDA(cudaMemsetAsync(s.probs.data, 0, elements * sizeof(float), stream));
#define UPLOAD(dst, vec) do { if (!(vec).empty()) CHECK_CUDA(cudaMemcpyAsync((dst).data, (vec).data(), (vec).size() * sizeof((vec)[0]), cudaMemcpyHostToDevice, stream)); } while (0)
    UPLOAD(s.seen_ids, r.seen); UPLOAD(s.drafts, r.drafts); UPLOAD(s.candidates, r.candidate_ids);
    UPLOAD(s.q, r.proposal_probabilities); UPLOAD(s.uniforms, r.uniforms);
    if (!r.seen.empty()) mark_seen<<<(r.seen.size() + 255) / 256, 256, 0, stream>>>(s.seen.data, s.seen_ids.data, r.seen.size(), r.vocab);
    penalize<<<(elements + 255) / 256, 256, 0, stream>>>(s.adjusted.data, s.ids.data, logits, s.seen.data, s.drafts.data, s.invalid.data, c);
    if (fast) {
        CHECK_CUDA(s.top_ids.reserve(uint64_t(rows) * parts * small_k)); CHECK_CUDA(s.top_values.reserve(uint64_t(rows) * parts * small_k));
        CHECK_CUDA(dflash_cuda::topk(s.ranked_ids.data, s.ranked.data, s.adjusted.data, s.top_ids.data, s.top_values.data,
                                     rows, r.vocab, small_k, stream));
    } else {
        offsets.resize(rows + 1);
        for (uint32_t i = 0; i <= rows; ++i) offsets[i] = i * r.vocab;
        CHECK_CUDA(s.offsets.reserve(offsets.size())); UPLOAD(s.offsets, offsets);
        size_t bytes = 0;
        CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, bytes, s.adjusted.data, s.ranked.data,
            s.ids.data, s.ranked_ids.data, int(elements), int(rows), s.offsets.data, s.offsets.data + 1, 0, 32, stream));
        CHECK_CUDA(s.sort.reserve(bytes));
        CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortPairsDescending(s.sort.data, bytes, s.adjusted.data, s.ranked.data,
            s.ids.data, s.ranked_ids.data, int(elements), int(rows), s.offsets.data, s.offsets.data + 1, 0, 32, stream));
    }
    normalizer<<<rows, 256, 0, stream>>>(s.info.data, s.adjusted.data, s.ranked.data, s.invalid.data, c);
    if (unfiltered) {
        full_softmax<<<(elements + 255) / 256, 256, 0, stream>>>(s.probs.data, s.adjusted.data, s.ranked.data, s.info.data, c);
        full_statistics<<<rows, 256, 0, stream>>>(s.info.data, s.probs.data, c);
    } else filter<<<rows, 1, 0, stream>>>(s.probs.data, s.info.data, s.ranked.data, s.ranked_ids.data, c);
    decide<<<1, 1, 0, stream>>>(s.answer.data, s.probs.data, s.info.data, s.drafts.data, s.candidates.data, s.q.data, s.uniforms.data, c);
    residual<<<(r.vocab + 255) / 256, 256, 0, stream>>>(s.weights.data, s.sums.data, s.probs.data, s.info.data, s.answer.data, s.candidates.data, s.q.data, c);
    draw<<<1, 1, 0, stream>>>(s.answer.data, s.weights.data, s.sums.data, s.uniforms.data, c);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpyAsync(&answer, s.answer.data, sizeof(answer), cudaMemcpyDeviceToHost, stream));
    result.probabilities.clear();
    result.row_fallbacks.clear();
    if (r.collect_probabilities) {
        report_info.resize(rows);
        result.probabilities.resize(elements);
        CHECK_CUDA(cudaMemcpyAsync(result.probabilities.data(), s.probs.data, elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaMemcpyAsync(report_info.data(), s.info.data, rows * sizeof(RowInfo), cudaMemcpyDeviceToHost, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    sync.done = true;
    result.accepted = answer.accepted; result.token = answer.token; result.rng_draws = answer.draws; result.fallback = answer.fallback;
    for (const auto &info : report_info) result.row_fallbacks.push_back(info.fallback);
    return cudaSuccess;
#undef UPLOAD
#undef CHECK_CUDA
}
} // namespace qw3
