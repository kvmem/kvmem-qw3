// Generated-text regression for the experimental refresh boundary controller.
// Synthetic inputs and shortened triggers exercise branches; timings do not
// qualify real DeepSWE performance. No commands in generated output execute.
#include "backend.hpp"
#include "qw3/gguf.hpp"
#include "qw3/tokenizer.hpp"
#include "token_piece_id_map.hpp"
#include "json.hpp"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace qw3;
using Json = nlohmann::json;

void require(bool value, const std::string &message) {
    if (!value) throw std::runtime_error(message);
}

struct Case {
    std::string name, question;
    uint32_t trigger = 13, epoch_limit = 32768, refreshes = 1, cancel_after = 0;
};

int main(int argc, char **argv) try {
    const bool diagnose_greedy = argc == 6 && std::string(argv[5]) == "--diagnose-greedy";
    require(argc == 5 || diagnose_greedy,
        "usage: PROBE TARGET DRAFT OUTPUT_JSONL HISTORY_TOKENS [--diagnose-greedy]");
    const uint32_t history_tokens = std::stoul(argv[4]);
    require(history_tokens == 4096 || history_tokens == 131072, "unfrozen history size");
    require(!std::ifstream(argv[3]).good(), "refusing to overwrite output");
    std::ofstream output(argv[3]);
    require(bool(output), "cannot open output");
    setenv("QW3_KV_DTYPE", "fp8", 1);
    setenv("QW3_KVMEM_PREFIX_CACHE", "1", 1);
    setenv("QW3_KVMEM_PREFIX_CACHE_TRACE", "1", 1);
    GgufFile model(argv[1]); QwenTokenizer tokenizer(model);
    const test::TokenPieceIdMap id_map(tokenizer);
    auto encode = [&](const std::string &text) {
        auto tokens = tokenizer.encode(text); return std::vector<uint32_t>(tokens.begin(), tokens.end());
    };
    auto history = encode("<|im_start|>user\nEngineering notes.\n");
    const auto line = encode("The function fetch_record retries three times and has a timeout of 500 milliseconds.\n");
    while (history.size() < history_tokens) history.insert(history.end(), line.begin(), line.end());
    history.resize(history_tokens);
    const std::string xml = "Return exactly one XML tool call and no surrounding prose. Its structure must be "
        "<tool_call><function=bash><parameter=command>COMMAND</parameter></function></tool_call>. "
        "COMMAND must be a python heredoc that assigns a literal list of all integers from 0 through 127 "
        "to values and prints their sum. Write every integer explicitly; do not use range or ellipses.";
    std::vector<Case> cases{
        {"prose", "Explain network retries in six complete sentences, mentioning both retry limit and timeout."},
        {"two-fences", "Give two fenced Python code blocks: first implement merge sort with explanatory comments, "
            "then write five unit tests for it. Include all function bodies. No prose outside the blocks.", 13, 32768, 2},
        {"xml", xml},
        {"emergency-open-xml", xml, 13, 768},
        {"cancel-open-xml", xml, 13, 32768, 1, 64},
        {"after-cancel-xml", xml},
        {"after-private-query", "State the retry limit and timeout from the engineering notes in one sentence.", 0},
    };
    // Keep the exact first two requests and their order. This reproduces the
    // first long-history policy discrepancy and adds an identical off/off
    // control before blaming the deferred controller. It is not qualification.
    if (diagnose_greedy) cases.resize(2);
    Json definitions = Json::array();
    for (const auto &item : cases) definitions.push_back({{"name", item.name}, {"question", item.question},
        {"trigger", item.trigger}, {"epoch_limit", item.trigger ? item.epoch_limit : 0},
        {"steady_trigger", item.trigger ? 64 : 0}, {"query_max_tokens", item.trigger ? 512 : 0},
        {"max_refreshes", item.trigger ? item.refreshes : 0},
        {"cancel_after", item.cancel_after}});
    output << Json{{"type", "metadata"}, {"history_tokens", history_tokens}, {"working_tokens", 65536},
        {"generation_reserve", 32768}, {"index", "gpu-mean-k"}, {"repository_blocks", 0},
        {"performance_qualified", false}, {"diagnose_greedy", diagnose_greedy},
        {"cases", definitions}}.dump() << std::endl;
    std::vector<std::vector<uint32_t>> greedy_reference, sampling_reference;
    for (unsigned iteration = 0; iteration < (diagnose_greedy ? 3u : 4u); ++iteration) {
        const unsigned variant = diagnose_greedy ? (iteration == 2 ? 1 : 0) : iteration;
        const bool sampling = variant >= 2, deferred = variant != 0, gpu_sampler = variant != 2;
        const std::string label = diagnose_greedy && iteration == 1 ? "greedy_off_repeat" :
            variant == 0 ? "greedy_off" : variant == 1 ? "greedy_on" :
            variant == 2 ? "sampling_cpu_on" : "sampling_gpu_on";
        setenv("QW3_DFLASH_KVMEM_DEFERRED_REFRESH_SPECULATE", deferred ? "1" : "0", 1);
        setenv("QW3_GPU_TARGET_SAMPLING", gpu_sampler ? "1" : "0", 1);
        EngineOptions engine; engine.model_path = argv[1]; engine.dflash_model_path = argv[2];
        engine.ctx_size = 1048576; engine.prefill_chunk = 2048;
        engine.kvmem_enabled = true; engine.kvmem_block_tokens = 32;
        engine.kvmem_budget = engine.kvmem_prefill_budget = 65536; engine.kvmem_gen_budget = 32768;
        engine.kvmem_sink_blocks = 32; engine.kvmem_recent_blocks = 164;
        engine.kvmem_cpu_bytes = 64ull << 30; engine.kvmem_gpu_memory_ratio = .44;
        engine.kvmem_index_placement = "gpu"; engine.kvmem_retrieval_method = "mean-k";
        engine.kvmem_query_conditioned = true; engine.kvmem_update_mode = "step";
        engine.kvmem_immutable_source_k = true; engine.kvmem_repository_blocks = 0;
        auto backend = make_qwen_native_backend(); backend->load(engine);
        for (size_t index = 0; index < cases.size(); ++index) {
            const auto &item = cases[index];
            GenerationOptions request;
            request.temperature = sampling ? 1 : 0; request.seed = 1000 + index;
            request.top_k = 20; request.top_p = .95; request.min_p = 0;
            request.presence_penalty = 0; request.repetition_penalty = 1;
            request.max_tokens = 768; request.ignore_eos = false;
            request.prompt_token_ids_override = history;
            request.kvmem_query_begin = history.size();
            auto question = encode("\n" + item.question);
            request.prompt_token_ids_override.insert(request.prompt_token_ids_override.end(), question.begin(), question.end());
            request.kvmem_query_end = request.prompt_token_ids_override.size();
            auto ending = encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
            request.prompt_token_ids_override.insert(request.prompt_token_ids_override.end(), ending.begin(), ending.end());
            request.kvmem_reselect_mode = KvMemReselectMode::Force;
            request.kvmem_middecode_trigger_tokens = item.trigger;
            // The recovery-only case disables mid-decode refresh altogether;
            // a steady trigger without a first trigger is an invalid request.
            request.kvmem_middecode_steady_trigger_tokens = item.trigger ? 64 : 0;
            request.kvmem_middecode_max_refreshes = item.trigger ? item.refreshes : 0;
            request.kvmem_middecode_query_max_tokens = item.trigger ? 512 : 0;
            request.kvmem_middecode_epoch_limit_tokens = item.trigger ? item.epoch_limit : 0;
            std::vector<uint32_t> ids; std::string joined;
            std::cout << "REGRESSION_BEGIN variant=" << label << " case=" << item.name << std::endl;
            const auto started = std::chrono::steady_clock::now();
            const auto response = backend->generate("", request, [&](const std::string &piece) {
                ids.push_back(id_map.recover(piece)); joined += piece;
                return !item.cancel_after || ids.size() < item.cancel_after;
            });
            const bool compared = variant == 1 || variant == 3 || (diagnose_greedy && iteration == 1);
            const auto &reference = sampling ? sampling_reference : greedy_reference;
            const bool exact = !compared || ids == reference.at(index);
            output << Json{{"type", "request"}, {"variant", label}, {"case", item.name},
                {"input_tokens", request.prompt_token_ids_override.size()}, {"token_ids", ids}, {"text", response},
                {"max_tokens", request.max_tokens}, {"cancel_after", item.cancel_after},
                {"compared", compared}, {"exact_reference", exact}, {"callback_bytes_match", joined == response},
                {"wall_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()}}
                .dump() << std::endl;
            require(joined == response && !ids.empty() && ids.size() <= unsigned(request.max_tokens), "invalid emitted sequence");
            require(!item.cancel_after || ids.size() == item.cancel_after, "cancellation point not reached");
            require(exact, label + "/" + item.name + ": token IDs changed against paired reference");
            if (variant == 0 && (!diagnose_greedy || iteration == 0)) greedy_reference.push_back(ids);
            if (variant == 2) sampling_reference.push_back(ids);
            std::cout << "REGRESSION_END variant=" << label << " case=" << item.name << " tokens=" << ids.size() << std::endl;
        }
    }
    output << Json{{"type", "result"}, {"status", "paired_outputs_match_coverage_requires_log_audit"}}.dump() << std::endl;
    return 0;
} catch (const std::exception &error) {
    std::cerr << "deferred refresh regression: " << error.what() << '\n'; return 1;
}
