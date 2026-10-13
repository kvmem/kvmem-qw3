#include "qw3/device_backend.hpp"
#include "target_sampling.hpp"
#include "speculative_sampling.hpp"
#include "gpu_target_sampling.hpp"
#include "json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <stdexcept>

using namespace qw3;
void check(DeviceStatus s) { if (!s.ok) throw std::runtime_error(s.message); }
void require(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
std::vector<float> distribution(const std::vector<float> &data, unsigned row, const DeviceSamplingRequest &r) {
    std::vector<float> logits(data.begin() + uint64_t(row) * r.vocab, data.begin() + uint64_t(row + 1) * r.vocab);
    auto seen = r.seen;
    for (unsigned i = 0; i < row; ++i) seen.push_back(r.drafts[i]);
    std::sort(seen.begin(), seen.end()); seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
    for (auto id : seen) if (id < r.vocab) {
        float &v = logits[id];
        if (r.repetition_penalty > 0 && r.repetition_penalty != 1)
            v = v >= 0 ? v / r.repetition_penalty : v * r.repetition_penalty;
        v -= r.presence_penalty;
    }
    if (r.temperature <= 0) {
        std::vector<float> p(r.vocab);
        p[std::max_element(logits.begin(), logits.end()) - logits.begin()] = 1;
        return p;
    }
    return detail::sampling_distribution(logits, r.temperature, r.top_p, r.top_k, r.min_p,
        r.top_k_first ? SamplingFilterOrder::TopKFirst : SamplingFilterOrder::Joint);
}
struct Totals { unsigned cases = 0, fallback = 0, unused_ambiguous_rows = 0; double max_error = 0, gpu_ms = 0, cpu_ms = 0; };
void verify(DeviceBackend &backend, const std::vector<float> &data, DeviceSamplingRequest r, Totals &totals,
            int expected_accepted = -1) {
    auto device = backend.tensor_f32(data.size(), "sampling-oracle-logits");
    check(backend.copy_bytes_from_host(*device, 0, data.data(), data.size() * sizeof(float)));
    r.collect_probabilities = true;
    DeviceSamplingResult result;
    auto start = std::chrono::steady_clock::now();
    check(backend.sample_target_rows(*device, r, result));
    totals.gpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    ++totals.cases;
    if (result.fallback) { ++totals.fallback; return; }
    start = std::chrono::steady_clock::now();
    std::vector<std::vector<float>> p;
    for (unsigned row = 0; row <= r.drafts.size(); ++row) {
        p.push_back(distribution(data, row, r));
        if (result.row_fallbacks.at(row)) {
            // This hypothetical prefix is never reached: decide() stops at the
            // first rejection. If reached, its flag forces the CPU policy.
            require(row > unsigned(result.accepted), "used ambiguous row did not request fallback");
            ++totals.unused_ambiguous_rows;
            continue;
        }
        double sum = 0;
        for (unsigned id = 0; id < r.vocab; ++id) {
            const float actual = result.probabilities[uint64_t(row) * r.vocab + id], expected = p.back()[id];
            if ((actual > 0) != (expected > 0)) {
                std::cerr << "support: case=" << totals.cases << " row=" << row << " id=" << id << " gpu=" << actual << " cpu=" << expected << '\n';
                throw std::runtime_error("GPU filter changed target support");
            }
            totals.max_error = std::max(totals.max_error, double(std::abs(actual - expected)));
            require(std::abs(actual - expected) < 2e-6, "GPU target probability differs from CPU");
            sum += actual;
        }
        require(std::abs(sum - 1) < 3e-6, "GPU target probabilities do not sum to one");
    }
    unsigned accepted = 0, draws = 0;
    int token = -1;
    for (; accepted < r.drafts.size(); ++accepted) {
        auto step = detail::sparse_rejection_step(p[accepted], r.candidate_ids.data() + accepted * r.candidates_per_row,
             r.proposal_probabilities.data() + accepted * r.candidates_per_row, r.candidates_per_row, r.drafts[accepted]);
        ++draws;
        if (!step.accepts(r.uniforms[accepted])) {
            token = step.residual_token(r.uniforms[accepted + 1]);
            const auto positives = std::count_if(step.residual.begin(), step.residual.end(), [](double x) { return x > 0; });
            draws += r.residual_always_draw || positives > 1;
            break;
        }
    }
    if (accepted == r.drafts.size()) {
        const auto &weights = p.back();
        double sum = 0; for (float w : weights) sum += w;
        const double target = sum * r.uniforms[accepted];
        double cumulative = 0;
        for (unsigned i = 0; i < r.vocab; ++i) if (weights[i] > 0) {
            token = i; cumulative += weights[i]; if (target < cumulative) break;
        }
        draws += std::count_if(weights.begin(), weights.end(), [](float x) { return x > 0; }) > 1;
    }
    totals.cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    require(result.accepted == int(accepted), "GPU rejection boundary differs from reference");
    require(result.token == token, "GPU residual/bonus CDF differs from reference");
    require(result.rng_draws == int(draws), "GPU request RNG consumption changed");
    if (expected_accepted >= 0) require(result.accepted == expected_accepted, "forced rejection boundary missing");
}
void verify_ragged_bridge(DeviceBackend &backend) {
    constexpr uint32_t vocab = 37, total_rows = 40;
    const uint32_t offsets[] = {3, 10, 24}, widths[] = {1, 4, 8};
    std::vector<float> data(total_rows * vocab, std::numeric_limits<float>::quiet_NaN());
    std::mt19937_64 data_rng(833);
    std::uniform_real_distribution<float> value(-7, 7);
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned r = offsets[i]; r < offsets[i] + widths[i]; ++r)
            for (unsigned v = 0; v < vocab; ++v) data[r * vocab + v] = value(data_rng);
    auto batch = backend.tensor_f32(data.size(), "sampling-ragged-matrix");
    check(backend.copy_bytes_from_host(*batch, 0, data.data(), data.size() * sizeof(float)));
    GenerationOptions options[3];
    std::unique_ptr<detail::GpuTargetSampling> samplers[3];
    std::mt19937_64 rngs[3];
    for (unsigned i = 0; i < 3; ++i) {
        options[i].temperature = .7f; options[i].top_k = 7; options[i].top_p = .93f;
        options[i].presence_penalty = .2f; options[i].repetition_penalty = 1.1f;
        samplers[i] = std::make_unique<detail::GpuTargetSampling>(backend, vocab, options[i]);
        rngs[i].seed(101 + i * 19);
    }
    unsigned cases = 0;
    for (unsigned round = 0; round < 8; ++round) for (unsigned index = 0; index < 3; ++index) {
        const unsigned i = round % 2 ? 2 - index : index;
        auto own = backend.tensor_f32(widths[i] * vocab, "sampling-request-copy");
        check(backend.copy_bytes_from_host(*own, 0, data.data() + offsets[i] * vocab,
                                          widths[i] * vocab * sizeof(float)));
        std::vector<uint32_t> drafts(widths[i] - 1, (round + i) % vocab);
        DFlashProposal proposal; proposal.top_k = 2;
        for (auto d : drafts) {
            proposal.candidates.push_back(d); proposal.candidates.push_back((d + 1) % vocab);
            proposal.probabilities.push_back(.6f); proposal.probabilities.push_back(.4f);
        }
        const auto *q = i == 2 ? &proposal : nullptr;
        const std::unordered_map<uint32_t, uint32_t> seen{{i, 1}, {i + 3, 2}};
        auto reference_rng = rngs[i];
        DeviceSamplingResult expected, actual;
        const bool reference_ok = samplers[i]->sample(own.get(), seen, reference_rng, expected, drafts, q);
        const bool actual_ok = samplers[i]->sample(batch.get(), seen, rngs[i], actual, drafts, q, offsets[i]);
        require(actual_ok && reference_ok, "ragged bridge unexpectedly fell back");
        require(actual.token == expected.token && actual.accepted == expected.accepted &&
                actual.rng_draws == expected.rng_draws && reference_rng == rngs[i],
                "ragged bridge changed request decision or RNG");
        ++cases;
    }
    const auto before = rngs[0];
    DeviceSamplingResult fallback;
    require(!samplers[0]->sample(batch.get(), {}, rngs[0], fallback, {}, nullptr, 1) &&
            fallback.fallback && before == rngs[0], "ragged fallback consumed RNG");
    bool threw = false;
    try { samplers[0]->sample(batch.get(), {}, rngs[0], fallback, {}, nullptr, UINT64_MAX); }
    catch (const std::runtime_error &) { threw = true; }
    require(threw && before == rngs[0], "invalid ragged row range consumed RNG or was accepted");
    std::cout << "ragged_sampling_bridge cases=" << cases << " fallback_rng=unchanged bounds=rejected status=pass\n";
}
int main(int argc, char **argv) try {
    const bool small = argc == 2 && std::string(argv[1]) == "--small";
    require(argc <= 2, "usage: qw3-gpu-target-sampling [--small|SAMPLING_FIXTURE.json]");
    auto backend = make_cuda_device_backend(LinearBackend::Cublas); check(backend->begin());
    verify_ragged_bridge(*backend);
    std::mt19937_64 rng(983);
    std::uniform_real_distribution<float> logit(-7, 7);
    std::uniform_real_distribution<double> unit(0, 1);
    Totals totals;
    for (unsigned vocab : (small ? std::vector<unsigned>{37} : std::vector<unsigned>{1, 37, 257, 4097, 248320}))
      for (unsigned rows : (small ? std::vector<unsigned>{8} : std::vector<unsigned>{1, 4, 8})) {
        std::vector<float> data(uint64_t(rows) * vocab);
        for (auto &v : data) v = logit(rng);
        DeviceSamplingRequest r; r.vocab = vocab;
        r.candidates_per_row = std::min(vocab, 16u);
        r.seen = {0, 5, 10, 19, vocab + 3};
        r.presence_penalty = .27f; r.repetition_penalty = 1.13f;
        for (unsigned i = 0; i + 1 < rows; ++i) {
            r.drafts.push_back(i % r.candidates_per_row);
            for (unsigned j = 0; j < r.candidates_per_row; ++j) {
                r.candidate_ids.push_back(j);
                r.proposal_probabilities.push_back(float(j + 1)); // Intentionally not normalized.
            }
        }
        for (unsigned i = 0; i <= rows; ++i) r.uniforms.push_back(unit(rng));
        for (bool order : {false, true}) for (float temperature : {0.f, .6f, 1.f, 2.f})
          for (int kind = 0; kind < 9; ++kind) {
            r.top_k_first = order; r.temperature = temperature;
            r.top_k = kind == 0 ? 0 : kind == 1 ? 1 : kind == 2 ? 20 : kind == 3 ? 31 : kind == 4 ? 32 : kind == 5 ? 180 : 0;
            r.top_p = kind == 0 ? 1 : .95f;
            r.min_p = kind == 6 ? .1f : kind == 7 ? .8f : kind == 8 ? 2 : 0;
            verify(*backend, data, r, totals);
          }
        std::cout << "distribution_matrix vocab=" << vocab << " rows=" << rows << " passed\n" << std::flush;
    }
    for (bool sparse : {false, true}) for (unsigned n = 1; n <= 7; ++n) for (unsigned reject = 0; reject <= n; ++reject) {
        DeviceSamplingRequest r; r.vocab = 17; r.candidates_per_row = 1;
        r.drafts.assign(n, 0); r.candidate_ids.assign(n, 0); r.proposal_probabilities.assign(n, 1);
        r.uniforms.assign(n + 2, .01); r.uniforms[reject] = .99; r.residual_always_draw = sparse;
        std::vector<float> logits((n + 1) * r.vocab, -100);
        for (unsigned row = 0; row <= n; ++row) { logits[row * r.vocab] = 0; logits[row * r.vocab + 1] = 0; }
        const auto before = totals.fallback;
        verify(*backend, logits, r, totals, reject);
        require(before == totals.fallback, "forced boundaries unexpectedly fell back");
    }
    // Equal-logit top-k, ambiguous nucleus boundary, NaN and +inf must request
    // the exact legacy CPU route, without invalid device accesses.
    for (unsigned kind = 0; kind < 5; ++kind) {
        DeviceSamplingRequest r; r.vocab = 37; r.uniforms = {.3, .4}; r.top_k = 20;
        std::vector<float> logits(37, 0);
        if (kind == 1) { r.top_k = 0; r.top_p = .5; logits.assign(37, -100); logits[0] = logits[1] = 0; }
        if (kind == 2) logits[0] = NAN;
        if (kind == 3) logits[0] = INFINITY;
        if (kind == 4) { r.top_k = 0; r.min_p = 2; }
        const auto before = totals.fallback;
        verify(*backend, logits, r, totals);
        require(totals.fallback == before + 1, "ambiguous row did not fall back");
    }
    require(totals.fallback < totals.cases / 4, "GPU path fell back on too much of the matrix");
    if (!small) {
        // Independent marginal oracle: proposals follow q, while the first
        // committed token must follow p after sparse rejection/residual draws.
        DeviceSamplingRequest r; r.vocab = 3; r.candidates_per_row = 3;
        r.candidate_ids = {0, 1, 2}; r.proposal_probabilities = {7, 2, 1};
        std::vector<float> logits;
        for (int row = 0; row < 2; ++row)
            for (float p : {.5f, .3f, .2f}) logits.push_back(std::log(p));
        auto tensor = backend->tensor_f32(logits.size(), "sampling-marginal");
        check(backend->copy_bytes_from_host(*tensor, 0, logits.data(), logits.size() * sizeof(float)));
        std::discrete_distribution<int> proposal{7, 2, 1};
        unsigned histogram[3] = {}, accepted = 0, fallbacks = 0;
        const unsigned samples = 6000;
        for (unsigned i = 0; i < samples; ++i) {
            r.drafts = {proposal(rng)}; r.uniforms = {unit(rng), unit(rng), unit(rng)};
            DeviceSamplingResult result; check(backend->sample_target_rows(*tensor, r, result));
            if (result.fallback) {
                ++fallbacks;
                const auto p = distribution(logits, 0, r);
                const auto step = detail::sparse_rejection_step(p, r.candidate_ids.data(),
                    r.proposal_probabilities.data(), 3, r.drafts[0]);
                result.accepted = step.accepts(r.uniforms[0]);
                if (!result.accepted) result.token = step.residual_token(r.uniforms[1]);
            }
            ++histogram[result.accepted ? r.drafts[0] : result.token]; accepted += result.accepted;
        }
        const double p[] = {.5, .3, .2};
        for (unsigned i = 0; i < 3; ++i)
            require(std::abs(double(histogram[i]) / samples - p[i]) < 6 * std::sqrt(p[i] * (1 - p[i]) / samples),
                    "GPU speculative marginal differs from target distribution");
        require(std::abs(double(accepted) / samples - .8) < .025, "GPU acceptance mass differs from min(p,q)");
        require(fallbacks < 6, "marginal oracle exercised too few GPU decisions");
        std::cout << "sparse_marginal samples=" << samples << " fallbacks=" << fallbacks << " counts=" << histogram[0] << ',' << histogram[1] << ',' << histogram[2]
                  << " acceptance=" << double(accepted) / samples << " status=pass\n";
    }
    if (argc == 2 && !small) {
        std::ifstream file(argv[1]);
        nlohmann::json fixture; file >> fixture;
        for (const auto &item : fixture.at("cases")) {
            const auto logits = item.at("logits").get<std::vector<float>>();
            DeviceSamplingRequest r; r.vocab = logits.size(); r.uniforms = {.27183, .3819};
            r.temperature = item.at("temperature"); r.top_p = item.at("top_p"); r.top_k = item.at("top_k");
            r.top_k_first = item.at("order") == "top-k-first";
            std::vector<float> expected(r.vocab, 0);
            for (const auto &entry : item.at("nonzero")) expected.at(entry[0].get<size_t>()) = entry[1].get<float>();
            const auto cpu = distribution(logits, 0, r);
            for (unsigned i = 0; i < r.vocab; ++i)
                require((cpu[i] > 0) == (expected[i] > 0) && std::abs(cpu[i] - expected[i]) < 2e-6,
                        "CPU compatibility policy differs from pinned upstream distribution");
            verify(*backend, logits, r, totals);
        }
        std::cout << "pinned_sglang_cases=" << fixture.at("cases").size() << " status=pass\n";
    }
    std::cout << "gpu_target_sampling cases=" << totals.cases << " compatibility_fallbacks=" << totals.fallback
              << " unused_ambiguous_rows=" << totals.unused_ambiguous_rows
              << " max_probability_error=" << totals.max_error << " gpu_oracle_ms=" << totals.gpu_ms
              << " cpu_oracle_ms=" << totals.cpu_ms << " status=pass\n";
    return 0;
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
