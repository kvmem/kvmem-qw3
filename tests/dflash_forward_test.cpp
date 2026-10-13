#include "dflash_executor.hpp"
#include "dflash_batch_executor.hpp"
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void check(qw3::DeviceStatus status) {
    if (!status.ok) throw std::runtime_error(status.message);
}
std::vector<float> read(const std::filesystem::path &path, size_t count) {
    std::ifstream file(path, std::ios::binary);
    if (!file || std::filesystem::file_size(path) != count * sizeof(float))
        throw std::runtime_error("invalid fixture tensor: " + path.string());
    std::vector<float> result(count);
    if (count && !file.read(reinterpret_cast<char *>(result.data()), count * sizeof(float)))
        throw std::runtime_error("cannot read fixture tensor");
    return result;
}
std::unique_ptr<qw3::DeviceTensor> upload(qw3::DeviceBackend &backend, const std::vector<float> &data) {
    auto result = backend.tensor_f32(data.size(), "dflash-fixture");
    check(backend.copy_bytes_from_host(*result, 0, data.data(), data.size() * sizeof(float)));
    return result;
}
} // namespace

int main(int argc, char **argv) try {
    if (argc != 2) throw std::runtime_error("usage: qw3-dflash-forward FIXTURE_DIRECTORY");
    const std::filesystem::path root(argv[1]);
    nlohmann::json fixture;
    std::ifstream(root / "fixture.json") >> fixture;
    qw3::DFlashModel model(root.string());
    auto backend = qw3::make_cuda_device_backend(qw3::LinearBackend::Cublas);
    check(backend->begin());
    qw3::DFlashWeights weights(model, *backend);
    qw3::DFlashExecutor executor(weights, *backend);
    const auto &c = model.config();
    const auto head_data = read(root / "lm_head.f32", size_t(c.hidden_size) * c.vocab_size);
    auto head = backend->weight_f32(head_data.data(), head_data.size(), "fixture-head");
    head->rows = c.vocab_size; head->cols = c.hidden_size;
    float worst = 0;
    uint32_t index = 0;
    const bool bf16 = fixture.value("weight_dtype", std::string("fp32")) == "bf16";
    for (const auto &item : fixture.at("cases")) {
        const uint32_t prefix = item.at("prefix"), block = item.at("block"), base = item.at("base");
        const uint32_t width = c.hidden_size * c.target_layer_ids.size();
        const auto features = read(root / item.at("features").get<std::string>(), size_t(prefix) * width);
        auto noise = upload(*backend, read(root / item.at("noise").get<std::string>(), size_t(block) * c.hidden_size));
        const auto expected = read(root / item.at("output").get<std::string>(), size_t(block) * c.hidden_size);
        for (bool chunked : {false, true}) {
            executor.reset(base);
            uint32_t done = 0;
            while (done < prefix) {
                const uint32_t rows = std::min(prefix - done, chunked ? 5u : prefix);
                std::vector<float> part(features.begin() + size_t(done) * width,
                                        features.begin() + size_t(done + rows) * width);
                auto source = upload(*backend, part);
                executor.append_context(*source, rows, base + done);
                // Source lifetime ends here: synchronize the fixture uploader;
                // production owns target feature buffers across the whole step.
                check(backend->synchronize());
                done += rows;
            }
            if (executor.context_rows() != std::min(prefix, c.sliding_window))
                throw std::runtime_error("draft context is not bounded");
            // Repeating a rejected proposal must not mutate committed history.
            for (unsigned repeat = 0; repeat < 2; ++repeat) {
                const auto &output = executor.forward(*noise, block, base + prefix);
                std::vector<float> actual(expected.size());
                check(backend->copy_to_host(output, actual.data(), 0, actual.size()));
                float error = 0;
                double squared_error = 0, squared_reference = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    if (!std::isfinite(actual[i])) throw std::runtime_error("non-finite draft output");
                    error = std::max(error, std::abs(actual[i] - expected[i]));
                    squared_error += std::pow(double(actual[i]) - expected[i], 2);
                    squared_reference += double(expected[i]) * expected[i];
                }
                const double relative_l2 = std::sqrt(squared_error / std::max(squared_reference, 1e-30));
                worst = std::max(worst, error);
                std::cout << "case=" << index << " prefix=" << prefix << " block=" << block
                          << " chunked=" << chunked << " repeat=" << repeat << " max_abs=" << error
                          << " relative_l2=" << relative_l2 << '\n';
                // CPU/GPU reduction order can straddle a BF16 rounding
                // midpoint at a later GEMM input, even with FP32 residuals.
                // Bound both worst-coordinate and relative error by BF16's
                // precision; FP32 fixtures retain their stricter bound.
                if (error > (bf16 ? 1.0f / 128 : 5e-5f) || relative_l2 > (bf16 ? 1.0 / 256 : 5e-5))
                    throw std::runtime_error("full draft differs from official topology/precision oracle");
                std::mt19937_64 rng(1);
                const auto proposal = executor.propose(*head, item.at("anchor"), 0, rng);
                if (proposal.tokens != item.at("greedy_tokens").get<std::vector<int32_t>>())
                    throw std::runtime_error("full draft greedy selector differs from official oracle");
                if (executor.next_position() != base + prefix)
                    throw std::runtime_error("proposal changed committed position");
            }
        }
        ++index;
    }
    std::cout << "full_draft_cases=" << index << " maximum_absolute_error=" << worst << " passed\n";
    // Distinct request caches share only the draft weights. Compare the joint
    // block with the frozen official topology oracle and check that proposals
    // leave every committed archive unchanged. Repeat after reversing slots.
    qw3::DFlashBatchExecutor batch_executor(weights, *backend);
    unsigned joint_checks = 0;
    for (uint32_t block : {1u, 3u, 8u}) for (uint32_t count : {1u, 4u, 16u}) {
        if (block > c.block_size) continue;
        std::vector<nlohmann::json> cases;
        for (const auto &item : fixture.at("cases"))
            if (item.at("block").get<uint32_t>() == block) cases.push_back(item);
        if (cases.empty()) throw std::runtime_error("fixture is missing joint draft block width");
        std::vector<std::unique_ptr<qw3::DFlashExecutor>> executors;
        std::vector<std::unique_ptr<qw3::DeviceTensor>> noises;
        std::vector<qw3::DFlashBatchExecutor::Request> requests;
        std::vector<std::mt19937_64> rngs(count);
        std::vector<std::vector<float>> expected;
        std::vector<qw3::KvMemArchiveState> before(count);
        for (uint32_t i = 0; i < count; ++i) {
            const auto &item = cases[i % cases.size()];
            const uint32_t prefix = item.at("prefix"), base = item.at("base");
            executors.push_back(std::make_unique<qw3::DFlashExecutor>(weights, *backend));
            auto &e = *executors.back(); e.reset(base);
            auto source = upload(*backend, read(root / item.at("features").get<std::string>(),
                size_t(prefix) * c.hidden_size * c.target_layer_ids.size()));
            if (prefix) e.append_context(*source, prefix, base);
            check(backend->synchronize());
            noises.push_back(upload(*backend, read(root / item.at("noise").get<std::string>(), size_t(block) * c.hidden_size)));
            expected.push_back(read(root / item.at("output").get<std::string>(), size_t(block) * c.hidden_size));
            rngs[i].seed(141 + i);
            requests.push_back({&e, noises.back().get(), block, uint64_t(base) + prefix,
                item.at("anchor").get<uint32_t>(), 0, &rngs[i]});
            before[i].position = e.next_position(); e.capture_archive_state(before[i]);
        }
        for (float temperature : {0.f, 1.f}) for (bool reverse : {false, true}) {
            for (auto &r : requests) r.temperature = temperature;
            auto scheduled = requests;
            if (reverse) std::reverse(scheduled.begin(), scheduled.end());
            const auto rng_before = rngs;
            const auto proposals = batch_executor.propose(scheduled, *head);
            std::vector<float> actual(size_t(count) * block * c.hidden_size);
            check(backend->copy_to_host(*batch_executor.output(), actual.data(), 0, actual.size()));
            float max_error = 0, max_probability_error = 0;
            for (uint32_t slot = 0; slot < count; ++slot) {
                const uint32_t i = reverse ? count - 1 - slot : slot;
                const auto &item = cases[i % cases.size()];
                auto &e = *executors[i];
                for (size_t j = 0; j < expected[i].size(); ++j) {
                    const float a = actual[size_t(slot) * expected[i].size() + j];
                    if (!std::isfinite(a)) throw std::runtime_error("non-finite joint draft output");
                    max_error = std::max(max_error, std::abs(a - expected[i][j]));
                }
                if (temperature == 0 && proposals[slot].tokens != item.at("greedy_tokens").get<std::vector<int32_t>>())
                    throw std::runtime_error("joint draft greedy selector differs from official oracle");
                // The batch publishes each request's own hidden block. Its
                // private selector must agree with the joint selector on that
                // same input, including conditional probabilities and RNG use.
                auto reference_rng = rng_before[i];
                const auto reference = e.propose(*head, requests[i].anchor, temperature, reference_rng);
                if (reference.tokens != proposals[slot].tokens || reference.candidates != proposals[slot].candidates || reference_rng != rngs[i])
                    throw std::runtime_error("joint selector mixed request candidates or RNG state");
                for (size_t j = 0; j < reference.probabilities.size(); ++j)
                    max_probability_error = std::max(max_probability_error,
                        std::abs(reference.probabilities[j] - proposals[slot].probabilities[j]));
                qw3::KvMemArchiveState after; after.position = e.next_position(); e.capture_archive_state(after);
                if (after.payload != before[i].payload || after.position != before[i].position)
                    throw std::runtime_error("joint draft changed committed cache");
                ++joint_checks;
            }
            if (max_error > (bf16 ? 1.f / 128 : 5e-5f) || max_probability_error > 2e-5f)
                throw std::runtime_error("joint draft output/probabilities exceed oracle tolerance");
            std::cout << "joint_draft blocks=" << count << " width=" << block << " reverse=" << reverse
                      << " temperature=" << temperature << " max_abs=" << max_error
                      << " max_probability_error=" << max_probability_error << " cache_unchanged=1 status=pass\n";
        }
    }
    std::cout << "joint_draft_request_checks=" << joint_checks << " status=pass\n";
    // Logical history exceeds both 32-bit positions and the trained context.
    // Compare each bounded cache with a fresh suffix-only oracle while the
    // live executor repeatedly rebases its independent draft RoPE frame.
    qw3::DFlashExecutor fresh(weights, *backend);
    const uint64_t origin = (uint64_t(1) << 40) + 123;
    const uint32_t width = c.hidden_size * c.target_layer_ids.size();
    const uint32_t total = c.max_position_embeddings * 3 + 17;
    std::vector<float> history(size_t(total) * width);
    for (size_t i = 0; i < history.size(); ++i) history[i] = std::sin(double(i) * .013);
    auto noise = upload(*backend, read(root / fixture.at("cases")[0].at("noise").get<std::string>(),
                                       size_t(c.block_size) * c.hidden_size));
    executor.reset(origin);
    auto empty = executor.snapshot();
    uint32_t rebases = 0, saved_checks = 0;
    for (uint32_t done = 0; done < total;) {
        const uint32_t rows = std::min(7u, total - done);
        const auto before_rope = executor.rope_next();
        auto input = upload(*backend, std::vector<float>(history.begin() + size_t(done) * width,
            history.begin() + size_t(done + rows) * width));
        executor.append_context(*input, rows, origin + done);
        check(backend->synchronize());
        done += rows;
        if (executor.rope_next() < before_rope) ++rebases;
        if (executor.rope_next() + c.block_size > c.max_position_embeddings)
            throw std::runtime_error("draft rebase did not bound RoPE positions");
        const uint32_t suffix = std::min(done, c.sliding_window);
        fresh.reset(origin + done - suffix);
        auto source = upload(*backend, std::vector<float>(history.begin() + size_t(done - suffix) * width,
            history.begin() + size_t(done) * width));
        fresh.append_context(*source, suffix, origin + done - suffix);
        std::vector<float> actual(size_t(c.block_size) * c.hidden_size), reference(actual.size());
        check(backend->copy_to_host(executor.forward(*noise, c.block_size, origin + done), actual.data(), 0, actual.size()));
        check(backend->copy_to_host(fresh.forward(*noise, c.block_size, origin + done), reference.data(), 0, reference.size()));
        float error = 0;
        for (size_t i = 0; i < actual.size(); ++i) error = std::max(error, std::abs(actual[i] - reference[i]));
        if (error > (bf16 ? 1.0f / 128 : 5e-5f))
            throw std::runtime_error("rebased cache differs from fresh suffix oracle");
        if (done % 49 == 0) {
            const auto saved = executor.snapshot();
            executor.reset(19);
            executor.restore(saved);
            std::vector<float> restored(actual.size());
            check(backend->copy_to_host(executor.forward(*noise, c.block_size, origin + done), restored.data(), 0, restored.size()));
            if (restored != actual) throw std::runtime_error("draft snapshot changed continuation bits");
            qw3::KvMemArchiveState archived;
            archived.position = executor.next_position();
            executor.capture_archive_state(archived);
            executor.reset(29);
            executor.restore_archive_state(archived);
            check(backend->copy_to_host(executor.forward(*noise, c.block_size, origin + done), restored.data(), 0, restored.size()));
            if (restored != actual) throw std::runtime_error("archived draft changed continuation bits");
            auto wrong = archived;
            const auto &identity = weights.identity_sha256();
            auto begin = std::search(wrong.payload.begin(), wrong.payload.end(), identity.begin(), identity.end());
            if (begin == wrong.payload.end()) throw std::runtime_error("archive has no draft identity");
            *begin = *begin == '0' ? '1' : '0';
            executor.restore_archive_state(wrong);
            if (executor.next_position() != origin + done || executor.context_rows() != 0 || executor.rope_next() != 0)
                throw std::runtime_error("different-draft archive did not cold-reset");
            executor.restore_archive_state(archived);
            qw3::KvMemArchiveState legacy;
            legacy.position = origin + done;
            executor.restore_archive_state(legacy);
            if (executor.next_position() != legacy.position || executor.context_rows() != 0)
                throw std::runtime_error("legacy archive retained stale draft cache");
            executor.restore_archive_state(archived);
            ++saved_checks;
        }
    }
    executor.restore(empty);
    if (executor.next_position() != origin || executor.context_rows() != 0 || executor.rope_next() != 0)
        throw std::runtime_error("empty draft snapshot restore failed");
    if (rebases < 2 || saved_checks < 2) throw std::runtime_error("fixture did not cover draft rebase/restore");
    std::cout << "logical_origin=" << origin << " rebases=" << rebases
              << " snapshot_continuations_exact=" << saved_checks << " bounded_history_passed\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}
