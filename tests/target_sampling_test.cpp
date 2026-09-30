#include "target_sampling.hpp"
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

using qw3::SamplingFilterOrder;
using qw3::detail::sampling_distribution;

namespace {
void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

// Full-sort, long-double oracle for the mathematical filtering contract.
std::vector<float> oracle(const std::vector<float> &logits, float temperature,
        float top_p, int top_k, float min_p, SamplingFilterOrder order) {
    std::vector<size_t> rank(logits.size());
    std::iota(rank.begin(), rank.end(), 0);
    std::sort(rank.begin(), rank.end(), [&](size_t a, size_t b) { return logits[a] > logits[b]; });
    const size_t k = top_k > 0 ? std::min(size_t(top_k), rank.size()) : rank.size();
    std::vector<long double> weights(rank.size());
    long double normalizer = 0;
    for (size_t i = 0; i < rank.size(); ++i) {
        weights[i] = std::exp((static_cast<long double>(logits[rank[i]]) - logits[rank[0]]) / temperature);
        if (order == SamplingFilterOrder::Joint || i < k) normalizer += weights[i];
    }
    long double cumulative = 0, retained = 0;
    size_t keep = 0;
    for (; keep < k; ++keep) {
        if (min_p > 0 && weights[keep] < weights[0] * min_p) break;
        retained += weights[keep];
        cumulative += weights[keep] / normalizer;
        if (top_p > 0 && top_p < 1 && cumulative >= top_p) { ++keep; break; }
    }
    std::vector<float> result(logits.size(), 0);
    if (keep == 0) { result[rank[0]] = 1; return result; }
    for (size_t i = 0; i < keep; ++i) result[rank[i]] = weights[i] / retained;
    return result;
}

void compare(const std::vector<float> &actual, const std::vector<float> &expected) {
    require(actual.size() == expected.size(), "distribution size mismatch");
    double sum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require((actual[i] > 0) == (expected[i] > 0), "candidate support mismatch");
        require(std::abs(actual[i] - expected[i]) < 2e-6, "candidate probability mismatch");
        sum += actual[i];
    }
    require(std::abs(sum - 1) < 2e-6, "distribution does not sum to one");
}
}

int main(int argc, char **argv) try {
    const std::vector<float> logits{std::log(.4f), std::log(.3f), std::log(.2f), std::log(.1f)};
    compare(sampling_distribution(logits, 1, .55f, 2, 0, SamplingFilterOrder::Joint),
        {4.f / 7, 3.f / 7, 0, 0});
    compare(sampling_distribution(logits, 1, .55f, 2, 0, SamplingFilterOrder::TopKFirst),
        {1, 0, 0, 0});
    compare(sampling_distribution({0, 0, 0, 0}, 1, .5f, 4, 0, SamplingFilterOrder::TopKFirst),
        {.25f, .25f, .25f, .25f});
    compare(sampling_distribution({0, 0, -100, -100}, 1, .5f, 4, 0, SamplingFilterOrder::TopKFirst),
        {.5f, .5f, 0, 0});
    require(sampling_distribution({}, 1, .95f, 20, 0, SamplingFilterOrder::Joint).empty(), "empty logits");
    std::mt19937 rng(183);
    std::uniform_real_distribution<float> value(-8, 8);
    size_t cases = 4;
    for (int n : {1, 37, 257, 2049}) {
        std::vector<float> row(n);
        for (float &v : row) v = value(rng);
        for (auto order : {SamplingFilterOrder::Joint, SamplingFilterOrder::TopKFirst})
            for (float temperature : {.6f, 1.f, 2.f})
                for (int k : {0, 1, 20, 180, 3000})
                    for (float p : {.55f, .95f, 1.f})
                        for (float min_p : {0.f, .1f, .8f, 2.f}) {
                            compare(sampling_distribution(row, temperature, p, k, min_p, order),
                                oracle(row, temperature, p, k, min_p, order));
                            ++cases;
                        }
    }
    std::cout << "target_sampling analytical_and_random_cases=" << cases << " status=pass\n";
    if (argc == 2) {
        std::ifstream file(argv[1]);
        nlohmann::json fixture;
        file >> fixture;
        size_t upstream_cases = 0;
        for (const auto &c : fixture.at("cases")) {
            std::vector<float> row = c.at("logits").get<std::vector<float>>();
            std::vector<float> expected(row.size(), 0);
            for (const auto &v : c.at("nonzero")) expected.at(v[0].get<size_t>()) = v[1].get<float>();
            const auto order = c.at("order") == "top-k-first"
                ? SamplingFilterOrder::TopKFirst : SamplingFilterOrder::Joint;
            compare(sampling_distribution(row, c.at("temperature"), c.at("top_p"), c.at("top_k"), 0, order), expected);
            ++upstream_cases;
        }
        std::cout << "pinned_sglang_cases=" << upstream_cases << " support=exact probability_abs_bound=2e-6 status=pass\n";
    }
    return 0;
} catch (const std::exception &error) {
    std::cerr << "target sampling failed: " << error.what() << '\n';
    return 1;
}
