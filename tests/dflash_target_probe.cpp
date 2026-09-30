#include "dflash_executor.hpp"
#include "qwen_executor.hpp"
#include "qw3/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

// Model-dependent integration diagnostic, deliberately excluded from CTest.
// Compares native greedy generation with and without the complete drafter.
// Its short prompts and timings are not a public performance benchmark.
namespace {
using namespace qw3;
void check(DeviceStatus status) {
    if (!status.ok) throw std::runtime_error(status.message);
}
void require(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error(message);
}
uint32_t prediction(const NativeExecutorReport &report) {
    std::string reason;
    for (const auto &s : report.missing_kernels) reason += " " + s;
    require(report.ok && report.argmax_token >= 0, "target forward failed:" + reason);
    return report.argmax_token;
}
void append(DFlashExecutor &draft, QwenExecutor &target, uint32_t rows) {
    const auto features = target.dflash_features();
    require(features.data && features.rows == rows &&
            features.logical_base == draft.next_position() &&
            features.rope_base == draft.next_position(), "target feature continuity lost");
    draft.append_context(*features.data, rows, features.rope_base);
}
uint32_t prefill(QwenExecutor &target, DFlashExecutor *draft, const std::vector<uint32_t> &prompt) {
    target.reset_state();
    if (draft) draft->reset();
    uint32_t next = 0;
    for (uint32_t offset = 0; offset < prompt.size();) {
        const uint32_t rows = target.effective_prefill_chunk_size(prompt.size() - offset);
        std::vector<uint32_t> part(prompt.begin() + offset, prompt.begin() + offset + rows);
        next = prediction(target.forward_n_tokens(part));
        if (draft) append(*draft, target, rows);
        offset += rows;
    }
    return next;
}

void rollback_oracle(QwenExecutor &target, DeviceBackend &backend,
                     const std::vector<uint32_t> &prompt,
                     const std::vector<uint32_t> &expected, const DFlashConfig &c) {
    prefill(target, nullptr, prompt);
    auto before = target.snapshot_state();
    QwenExecutor::StateCheckpointSet checkpoints;
    for (uint32_t keep = 1; keep < c.block_size; ++keep) {
        std::vector<float> reference_features;
        std::vector<std::vector<float>> reference_logits;
        for (bool corrupt : {false, true}) {
            target.restore_state(before);
            std::vector<uint32_t> tokens(expected.begin(), expected.begin() + c.block_size);
            if (corrupt) for (size_t i = keep; i < tokens.size(); ++i)
                tokens[i] = (tokens[i] + 1001 + i * 17) % c.vocab_size;
            std::vector<DeviceArgmax> next;
            prediction(target.forward_n_tokens(tokens, true, &next, &checkpoints, c.block_size - 1, false));
            target.restore_state_checkpoint(checkpoints, keep - 1);
            const auto features = target.dflash_features();
            require(features.data && features.rows == keep && target.position() == prompt.size() + keep,
                    "rollback retained rejected rows or lost position");
            std::vector<float> captured(uint64_t(keep) * c.hidden_size * c.target_layer_ids.size());
            check(backend.copy_to_host(*features.data, captured.data(), 0, captured.size()));
            if (!corrupt) reference_features = captured;
            else require(captured == reference_features, "rejected suffix changed accepted target features");
            for (uint32_t i = 0; i < 3; ++i) {
                prediction(target.forward_one_token(expected[keep + i]));
                std::vector<float> logits;
                require(target.copy_last_logits(logits), "rollback continuation logits missing");
                if (!corrupt) reference_logits.push_back(std::move(logits));
                else require(logits == reference_logits[i], "rejected suffix leaked into target continuation");
            }
        }
        std::cout << "rollback_keep=" << keep << " features=exact continuation_logits=exact\n" << std::flush;
    }
}

void batch_control(QwenExecutor &target, const std::vector<uint32_t> &prompt,
                   const std::vector<uint32_t> &expected,
                   const std::vector<std::vector<float>> &oracle) {
    prefill(target, nullptr, prompt);
    uint32_t mismatches = 0;
    float worst = 0;
    QwenExecutor::StateCheckpointSet checkpoints;
    for (size_t offset = 0; offset < expected.size(); offset += 8) {
        const auto rows = std::min<size_t>(8, expected.size() - offset);
        std::vector<uint32_t> tokens(expected.begin() + offset, expected.begin() + offset + rows);
        std::vector<DeviceArgmax> next;
        std::vector<std::vector<float>> logits;
        prediction(target.forward_n_tokens(tokens, true, &next, &checkpoints, rows - 1, false, &logits));
        for (size_t r = 0; r < rows && offset + r + 1 < expected.size(); ++r) {
            const auto index = offset + r + 1;
            mismatches += next[r].token != int32_t(expected[index]);
            for (size_t v = 0; v < logits[r].size(); ++v)
                worst = std::max(worst, std::abs(logits[r][v] - oracle[index][v]));
        }
    }
    std::cout << "target_only_teacher_forced_batch8 mismatches=" << mismatches
              << " logits_max_abs=" << worst << " (no draft calls or rejected rows)\n" << std::flush;
}

struct ReplayBlock {
    std::vector<uint32_t> input;
    uint32_t keep;
    std::vector<float> next_logits;
};
void replay_without_draft(QwenExecutor &target, const std::vector<uint32_t> &prompt,
                           const std::vector<ReplayBlock> &blocks) {
    prefill(target, nullptr, prompt);
    QwenExecutor::StateCheckpointSet checkpoints;
    for (const auto &block : blocks) {
        std::vector<DeviceArgmax> next;
        std::vector<std::vector<float>> rows;
        prediction(target.forward_n_tokens(block.input, true, &next, &checkpoints,
            block.input.size() - 1, false, &rows));
        if (block.keep < block.input.size()) target.restore_state_checkpoint(checkpoints, block.keep - 1);
        require(rows[block.keep - 1] == block.next_logits,
                "pure-target replay differs from target execution with draft calls");
    }
    std::cout << "target_only_matching_blocks=" << blocks.size()
              << " logits=exact (reproduces divergence without running the drafter)\n" << std::flush;
}

void paired_snapshot_oracle(QwenExecutor &target, DFlashExecutor &draft,
                            DeviceBackend &backend, const QwenWeights &weights,
                            const std::vector<uint32_t> &prompt,
                            const std::vector<uint32_t> &continuation) {
    const auto &c = draft.config();
    prefill(target, &draft, prompt);
    auto state = target.snapshot_state();
    require(state.dflash && state.dflash->next_position == state.position,
            "target snapshot omitted synchronized draft state");
    auto noise = backend.scratch_f32(uint64_t(c.block_size) * c.hidden_size, "paired-noise");
    const auto draft_output = [&]() {
        std::vector<uint64_t> ids(c.block_size, c.mask_token_id);
        ids.front() = continuation.front();
        check(backend.q8_0_get_rows_batch(*noise, weights.token_embd(), ids.data(), ids.size()));
        const auto &hidden = draft.forward(*noise, c.block_size, target.position());
        std::vector<float> result(uint64_t(c.block_size) * c.hidden_size);
        check(backend.copy_to_host(hidden, result.data(), 0, result.size()));
        return result;
    };
    const auto before = draft_output();
    std::vector<std::vector<float>> reference_logits;
    for (uint32_t branch = 0; branch < 2; ++branch) {
        target.restore_state(state);
        require(draft.next_position() == target.position() && draft.context_rows() == state.dflash->rows,
                "paired restore lost draft position or context");
        require(draft_output() == before, "paired restore changed draft output");
        for (uint32_t i = 0; i < 3; ++i) {
            prediction(target.forward_one_token(continuation[i]));
            append(draft, target, 1);
            std::vector<float> logits;
            require(target.copy_last_logits(logits), "paired continuation logits missing");
            if (branch == 0) reference_logits.push_back(std::move(logits));
            else require(logits == reference_logits[i], "paired restore changed target continuation");
        }
    }
    // Archive/older target-only snapshots have no draft history. Restoration
    // must reset at the logical target position, never retain a different branch.
    auto saved_draft = state.dflash;
    state.dflash.reset();
    target.restore_state(state);
    require(draft.next_position() == target.position() && draft.context_rows() == 0,
            "target-only restore retained stale draft context");
    state.dflash = std::move(saved_draft);
    target.restore_state(state);
    require(draft_output() == before, "target-only warmup damaged a saved draft state");
    std::cout << "paired_snapshot target_logits=exact draft_hidden=exact target_only_restore=empty\n" << std::flush;
}
} // namespace

int main(int argc, char **argv) try {
    if (argc < 3 || argc > 5)
        throw std::invalid_argument("usage: qw3-dflash-target-probe TARGET DRAFT [TOKENS=64] [TASK_INDEX]");
    const uint32_t count = argc >= 4 ? std::stoul(argv[3]) : 64;
    const int task_only = argc == 5 ? std::stoi(argv[4]) : -1;
    require(task_only >= -1 && task_only < 3, "invalid task index");
    require(count >= 16 && count <= 512, "probe token count outside 16..512");
    auto backend = make_cuda_device_backend(LinearBackend::Auto);
    check(backend->begin());
    QwenNativeModel model(open_model_source(argv[1]));
    DFlashModel draft_model(argv[2]);
    const auto &c = draft_model.config();
    c.validate_target(model.config().n_layers, model.config().n_embd, model.config().vocab_size);
    QwenWeights weights(model, *backend, false, false);
    DFlashWeights draft_weights(draft_model, *backend);
    DFlashExecutor draft(draft_weights, *backend);
    QwenExecutor target(model, weights, *backend, 4096);
    auto tokenizer = std::filesystem::is_directory(argv[1])
        ? std::make_unique<QwenTokenizer>(std::string(argv[1]))
        : std::make_unique<QwenTokenizer>(model.gguf());
    auto noise = backend->scratch_f32(uint64_t(c.block_size) * c.hidden_size, "draft-noise");
    std::mt19937_64 rng(101);
    const std::vector<std::string> questions{
        "Compute 17 * 23 and explain the calculation.",
        "Write a Python function that checks whether an integer is prime.",
        "用三句话解释为什么天空看起来是蓝色。"};
    for (size_t task = 0; task < questions.size(); ++task) {
        if (task_only >= 0 && task != size_t(task_only)) continue;
        const auto encoded = tokenizer->encode("<|im_start|>user\n" + questions[task] +
            "<|im_end|>\n<|im_start|>assistant\n<think>\n");
        const std::vector<uint32_t> prompt(encoded.begin(), encoded.end());
        target.attach_dflash(nullptr);
        uint32_t current = prefill(target, nullptr, prompt);
        std::vector<uint32_t> expected;
        std::vector<std::vector<float>> oracle_logits;
        for (uint32_t i = 0; i < count; ++i) {
            expected.push_back(current);
            oracle_logits.emplace_back();
            require(target.copy_last_logits(oracle_logits.back()), "missing serial logits");
            current = prediction(target.forward_one_token(current));
        }
        batch_control(target, prompt, expected, oracle_logits);
        target.set_dflash_layers(c.target_layer_ids);
        if (task == 0) rollback_oracle(target, *backend, prompt, expected, c);
        target.attach_dflash(&draft);
        if (task == 0) paired_snapshot_oracle(target, draft, *backend, weights, prompt, expected);
        current = prefill(target, &draft, prompt);
        std::vector<float> anchor_logits;
        require(target.copy_last_logits(anchor_logits), "missing anchor logits");
        std::vector<uint32_t> actual;
        std::vector<ReplayBlock> replay;
        uint32_t cycles = 0, rejected = 0;
        QwenExecutor::StateCheckpointSet checkpoints;
        const auto start = std::chrono::steady_clock::now();
        while (actual.size() < count) {
            const uint32_t rows = std::min<uint32_t>(c.block_size, count - actual.size());
            std::vector<uint64_t> ids(rows, c.mask_token_id);
            ids.front() = current;
            check(backend->begin());
            check(backend->q8_0_get_rows_batch(*noise, weights.token_embd(), ids.data(), rows));
            draft.forward(*noise, rows, draft.next_position());
            const auto proposal = draft.propose(weights.output(), current, 0, rng);
            std::vector<uint32_t> input{current};
            input.insert(input.end(), proposal.tokens.begin(), proposal.tokens.end());
            std::vector<DeviceArgmax> next;
            std::vector<std::vector<float>> row_logits;
            prediction(target.forward_n_tokens(input, true, &next, &checkpoints, rows - 1, false, &row_logits));
            require(next.size() == rows, "verifier lost rows");
            uint32_t accepted = 1;
            while (accepted < rows && next[accepted - 1].token == int32_t(input[accepted])) ++accepted;
            if (accepted < rows) target.restore_state_checkpoint(checkpoints, accepted - 1);
            append(draft, target, accepted);
            replay.push_back({input, accepted, row_logits[accepted - 1]});
            rejected += rows - accepted;
            for (uint32_t i = 0; i < accepted; ++i) {
                const auto index = actual.size();
                if (input[i] != expected[index]) {
                    const auto &p = i ? row_logits[i - 1] : anchor_logits;
                    const auto &reference = oracle_logits[index];
                    float maximum_error = 0;
                    for (size_t j = 0; j < p.size(); ++j)
                        maximum_error = std::max(maximum_error, std::abs(p[j] - reference[j]));
                    std::cerr << "task=" << task << " token=" << index << " expected=" << expected[index]
                              << " actual=" << input[i] << " serial_expected=" << reference[expected[index]]
                              << " serial_actual=" << reference[input[i]] << " verify_expected=" << p[expected[index]]
                              << " verify_actual=" << p[input[i]] << " logits_max_abs=" << maximum_error << "\n";
                    replay_without_draft(target, prompt, replay);
                    throw std::runtime_error("DFlash target output differs from serial greedy oracle");
                }
                actual.push_back(input[i]);
            }
            current = next[accepted - 1].token;
            anchor_logits = std::move(row_logits[accepted - 1]);
            ++cycles;
            require(target.position() == prompt.size() + actual.size() &&
                    draft.next_position() == target.position(), "commit positions diverged");
        }
        std::cout << "task=" << task << " tokens=" << actual.size() << " cycles=" << cycles
                  << " accepted_per_cycle=" << double(actual.size()) / cycles << " rejected=" << rejected
                  << " decode_ms=" << std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - start).count() << " greedy=exact\n";
        std::vector<int32_t> text(actual.begin(), actual.end());
        std::cout << tokenizer->decode(text) << '\n' << std::flush;
    }
    check(backend->end());
    std::cout << "DFlash target integration passed; public benchmark not measured\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << "DFlash target probe failed: " << e.what() << '\n';
    return 1;
}
