#include "backend.hpp"
#include "qw3/gguf.hpp"
#include "qw3/tokenizer.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>

// Real-model production-route regression. Deliberately excluded from CTest:
// requires the target + draft checkpoints and a GPU large enough to hold them.
// Diagnostic timings are not public benchmark results.
namespace {
void require(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error(message);
}
struct Case {
    std::string name;
    qw3::GenerationOptions request;
    uint32_t cancel_after = 0;
    bool require_full_length = true;
};
struct Result { std::string text; uint32_t callbacks; std::vector<std::string> pieces; };
}

int main(int argc, char **argv) try {
    if (argc != 4 && argc != 5) throw std::invalid_argument(
        "usage: qw3-dflash-backend-probe TARGET DRAFT dense|kvmem|kvmem-bounded [--plain-timing|--timing]");
    const bool plain_timing = argc == 5 && std::string(argv[4]) == "--plain-timing";
    const bool timing = plain_timing || (argc == 5 && std::string(argv[4]) == "--timing");
    require(argc == 4 || timing, "unknown probe option");
    const std::string mode = argv[3];
    require(mode == "dense" || mode == "kvmem" || mode == "kvmem-bounded", "unknown probe mode");
    const bool kvmem = mode != "dense";
    require(!timing || kvmem, "timing audit requires KVMem");
    setenv("QW3_KV_DTYPE", "fp16", 0);
    setenv("QW3_KVMEM_PREFIX_CACHE", "1", 1);
    setenv("QW3_KVMEM_PREFIX_CACHE_TRACE", "1", 1);
    setenv("QW3_KVMEM_DECODE_CHECKPOINT_TOKENS", "32", 1);
    std::unique_ptr<qw3::GgufFile> gguf;
    std::unique_ptr<qw3::QwenTokenizer> tokenizer;
    if (std::filesystem::is_directory(argv[1]))
        tokenizer = std::make_unique<qw3::QwenTokenizer>(std::string(argv[1]));
    else {
        gguf = std::make_unique<qw3::GgufFile>(argv[1]);
        tokenizer = std::make_unique<qw3::QwenTokenizer>(*gguf);
    }
    auto request = [&](const std::string &text, int tokens) {
        qw3::GenerationOptions r;
        r.temperature = 0; r.max_tokens = tokens; r.seed = 101; r.ignore_eos = true;
        const auto ids = tokenizer->encode(text);
        r.prompt_token_ids_override.assign(ids.begin(), ids.end());
        return r;
    };
    const std::string prompt = "<|im_start|>user\nCompute 17 * 23 and explain the calculation."
        "<|im_end|>\n<|im_start|>assistant\n<think>\n";
    std::vector<Case> cases;
    if (!kvmem) {
        for (int count : {0, 1, 2, 7, 8, 9, 64})
            cases.push_back({"limit-" + std::to_string(count), request(prompt, count)});
        auto penalties = request(prompt, 32);
        penalties.presence_penalty = 0.7f; penalties.repetition_penalty = 1.05f;
        cases.push_back({"greedy-penalties", penalties});
        for (int limit : {1, 7, 9}) {
            auto thinking = request(prompt, 64);
            thinking.thinking_open = true; thinking.thinking_budget = limit;
            cases.push_back({"thinking-budget-" + std::to_string(limit), thinking});
        }
    } else {
        cases.push_back({"identity", request(prompt, 32)});
        std::string history = "<|im_start|>user\n";
        for (int i = 0; i < 40; ++i) history += "Record " + std::to_string(i) +
            ": the retry limit is three and the timeout is 500 milliseconds.\n";
        cases.push_back({"history-prefill", request(history, 0)});
        auto query = request(history + "What are the retry limit and timeout?"
            "<|im_end|>\n<|im_start|>assistant\n<think>\n", 32);
        query.kvmem_query_begin = cases.back().request.prompt_token_ids_override.size();
        query.kvmem_query_end = query.prompt_token_ids_override.size();
        query.kvmem_reselect_mode = qw3::KvMemReselectMode::Force;
        cases.push_back({"sparse-query", query});
        cases.push_back({"sparse-query-repeat", query});
        auto append = request(history + "Record 40: the new retry limit is four.\n"
            "What is the new retry limit?"
            "<|im_end|>\n<|im_start|>assistant\n<think>\n", 32);
        append.kvmem_query_begin = query.kvmem_query_begin;
        append.kvmem_query_end = append.prompt_token_ids_override.size();
        append.kvmem_reselect_mode = qw3::KvMemReselectMode::Force;
        cases.push_back({"sparse-tail-rewrite", append});
    }
    auto eos_request = request("<|im_start|>user\nReply with just the word OK."
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", 64);
    eos_request.ignore_eos = false;
    cases.push_back({"stop-eos", eos_request, 0, false});
    cases.push_back({"cancel-after-3", request(prompt, 32), 3});
    cases.push_back({"after-cancel", request(prompt, 32)});
    auto sample = request(prompt, 32);
    sample.temperature = 1; sample.top_p = 0.95f; sample.top_k = 20;
    cases.push_back({"sample-first", sample});
    cases.push_back({"sample-repeat", sample});
    eos_request.temperature = 1;
    cases.push_back({"sample-eos-first", eos_request, 0, false});
    cases.push_back({"sample-eos-repeat", eos_request, 0, false});
    sample.sampling_filter_order = qw3::SamplingFilterOrder::TopKFirst;
    cases.push_back({"sample-top-k-first", sample});
    cases.push_back({"sample-top-k-repeat", sample});
    sample.presence_penalty = .7f; sample.repetition_penalty = 1.05f;
    cases.push_back({"sample-top-k-penalties-first", sample});
    cases.push_back({"sample-top-k-penalties-repeat", sample});
    if (timing) {
        // Longer timing audit; --plain-timing loads no draft, --timing checks
        // both routes against identical greedy output. Keep the exact
        // regression history/query and record first selection separately from
        // three repeats that can restore the query-boundary checkpoint.
        auto identity = cases[0]; identity.request.max_tokens = 256;
        auto history = cases[1];
        auto query = cases[2]; query.request.max_tokens = 256;
        cases = {identity, history, query};
        for (int i = 1; i <= 3; ++i) {
            query.name = "sparse-query-warm-" + std::to_string(i);
            cases.push_back(query);
        }
    }
    std::vector<Result> reference;
    for (bool use_draft : {false, true}) {
        if (plain_timing && use_draft) break;
        qw3::EngineOptions engine;
        engine.model_path = argv[1]; engine.ctx_size = 4096; engine.prefill_chunk = 128;
        if (use_draft) engine.dflash_model_path = argv[2];
        if (kvmem) {
            engine.kvmem_enabled = true; engine.kvmem_block_tokens = 32;
            engine.kvmem_budget = 256; engine.kvmem_gen_budget = 256;
            engine.kvmem_sink_blocks = 1; engine.kvmem_recent_blocks = 2;
            engine.kvmem_cpu_bytes = 2ull << 30; engine.kvmem_index_placement = "cpu";
            engine.kvmem_query_conditioned = true; engine.kvmem_update_mode = "step";
            if (mode == "kvmem-bounded") engine.kvmem_repository_blocks = 128;
        }
        auto backend = qw3::make_qwen_native_backend();
        backend->load(engine);
        std::string previous_sample;
        for (size_t index = 0; index < cases.size(); ++index) {
            const auto &test = cases[index];
            uint32_t callbacks = 0;
            std::vector<std::string> pieces;
            const auto start = std::chrono::steady_clock::now();
            auto first_token = start, last_token = start;
            auto text = backend->generate("", test.request, [&](const std::string &piece) {
                last_token = std::chrono::steady_clock::now();
                if (callbacks == 0) first_token = last_token;
                pieces.push_back(piece);
                ++callbacks;
                return !test.cancel_after || callbacks < test.cancel_after;
            });
            const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const auto expected_callbacks = test.cancel_after ? test.cancel_after : test.request.max_tokens;
            require(test.require_full_length ? callbacks == expected_callbacks : callbacks < expected_callbacks,
                test.name + ": incorrect token callback count or missing EOS stop");
            if (test.request.temperature > 0 && test.name.find("repeat") != std::string::npos)
                require(text == previous_sample,
                test.name + ": fixed-seed same-route output changed");
            previous_sample = text;
            if (!use_draft) reference.push_back({text, callbacks, std::move(pieces)});
            else if (test.request.temperature == 0) {
                if (text != reference[index].text) {
                    size_t first = 0;
                    while (first < pieces.size() && first < reference[index].pieces.size() &&
                           pieces[first] == reference[index].pieces[first]) ++first;
                    std::cerr << "parity_failure case=" << test.name << " first_callback=" << first
                              << " expected_text=" << std::quoted(reference[index].text)
                              << " actual_text=" << std::quoted(text) << '\n';
                }
                require(text == reference[index].text && callbacks == reference[index].callbacks,
                    test.name + ": DFlash output differs from serial target");
            }
            std::cout << "route=" << (use_draft ? "dflash" : "plain") << " mode=" << mode
                      << " case=" << test.name << " callbacks=" << callbacks
                      << " wall_s=" << seconds;
            if (callbacks) std::cout << " ttft_s="
                << std::chrono::duration<double>(first_token - start).count();
            if (callbacks > 1) std::cout << " callback_span_s="
                << std::chrono::duration<double>(last_token - first_token).count();
            // DFlash callbacks may arrive in bursts; callback_span_s is an
            // external latency diagnostic, not a per-kernel decode timer.
            std::cout << " status=pass\n" << std::flush;
        }
    }
    return 0;
} catch (const std::exception &error) {
    std::cerr << "DFlash backend probe failed: " << error.what() << '\n';
    return 1;
}
