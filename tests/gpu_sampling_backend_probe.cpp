#include "backend.hpp"
#include "qw3/gguf.hpp"
#include "qw3/tokenizer.hpp"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

// Same-route CPU/GPU seed parity; this is stricter than cross-algorithm
// stochastic quality and does not substitute for the frozen quality matrix.
int main(int argc, char **argv) try {
    using namespace qw3;
    if (argc < 3 || argc > 5) throw std::runtime_error("usage: qw3-gpu-sampling-backend-probe TARGET DRAFT [none|mtp|dflash|all] [--timing|--unfiltered-only|--cache-audit]");
    const std::string route = argc >= 4 ? argv[3] : "all";
    const bool timing = argc == 5 && std::string(argv[4]) == "--timing";
    const bool unfiltered_only = argc == 5 && std::string(argv[4]) == "--unfiltered-only";
    const bool cpu_cache_audit = argc == 5 && std::string(argv[4]) == "--cpu-cache-audit";
    const bool cache_audit = cpu_cache_audit || (argc == 5 && std::string(argv[4]) == "--cache-audit");
    if (argc == 5 && !timing && !unfiltered_only && !cache_audit) throw std::runtime_error("unknown flag");
    if (route != "all" && route != "none" && route != "mtp" && route != "dflash") throw std::runtime_error("unknown route");
    setenv("QW3_KV_DTYPE", "fp8", 1);
    setenv("QW3_KVMEM_PREFIX_CACHE", timing || cache_audit ? "1" : "0", 1);
    if (cpu_cache_audit) {
        setenv("QW3_KVMEM_TRACE", "1", 1);
        setenv("QW3_KVMEM_PREFIX_CACHE_TRACE", "1", 1);
    }
    GgufFile file(argv[1]); QwenTokenizer tokenizer(file);
    const auto encode = [&](const std::string &s) { auto ids = tokenizer.encode(s); return std::vector<uint32_t>(ids.begin(), ids.end()); };
    auto history = encode("<|im_start|>user\nEngineering log.\n");
    const auto line = encode("Function fetch_record retries network requests. The retry limit is three and the timeout is 500 milliseconds. Error: connection reset by peer.\n");
    while (history.size() < 4096) history.insert(history.end(), line.begin(), line.end());
    history.resize(4096);
    GenerationOptions base;
    base.max_tokens = 64; base.ignore_eos = true; base.seed = 101; base.temperature = 1;
    base.top_k = 20; base.top_p = .95f; base.sampling_filter_order = SamplingFilterOrder::TopKFirst;
    base.prompt_token_ids_override = history;
    base.kvmem_query_begin = history.size();
    auto query = encode("What is the function name, retry limit, timeout and network error? Explain how you found the evidence.");
    base.prompt_token_ids_override.insert(base.prompt_token_ids_override.end(), query.begin(), query.end());
    base.kvmem_query_end = base.prompt_token_ids_override.size();
    query = encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    base.prompt_token_ids_override.insert(base.prompt_token_ids_override.end(), query.begin(), query.end());
    base.kvmem_reselect_mode = KvMemReselectMode::Force;
    std::vector<std::pair<std::string, GenerationOptions>> cases;
    cases.push_back({"top-k-first", base});
    auto test = base; test.seed = 842; cases.push_back({"seed-842", test});
    test = base; test.sampling_filter_order = SamplingFilterOrder::Joint; cases.push_back({"joint", test});
    test = base; test.presence_penalty = .7; test.repetition_penalty = 1.13; test.min_p = .1; cases.push_back({"penalties", test});
    test = base; test.top_k = 0; cases.push_back({"full-nucleus", test});
    test = base; test.top_k = 0; test.top_p = 1; cases.push_back({"unfiltered", test});
    test = base; test.top_k = 180; test.temperature = .6; cases.push_back({"large-k", test});
    test = base; test.temperature = 0; test.presence_penalty = .7; test.repetition_penalty = 1.13; cases.push_back({"greedy-penalties", test});
    if (timing || cache_audit) {
        cases.clear(); base.max_tokens = 256;
        for (unsigned i = 0; i < 4; ++i) cases.push_back({i ? "warm-" + std::to_string(i) : "warmup", base});
    }
    if (unfiltered_only) cases = {{"unfiltered", cases[5].second}};
    unsigned passed = 0;
    for (const std::string mode : {"none", "mtp", "dflash"}) {
        if (route != "all" && route != mode) continue;
        EngineOptions e; e.model_path = argv[1]; e.ctx_size = 8192; e.prefill_chunk = 256;
        e.native_mtp_chain_set = true; e.native_mtp_chain = mode == "mtp" ? 4 : 0;
        e.native_mtp_speculate = mode == "mtp";
        if (mode == "dflash") { e.native_mtp_chain_set = false; e.dflash_model_path = argv[2]; }
        e.kvmem_enabled = true; e.kvmem_block_tokens = 32; e.kvmem_budget = 1024; e.kvmem_gen_budget = 512;
        e.kvmem_sink_blocks = 1; e.kvmem_recent_blocks = 4; e.kvmem_cpu_bytes = 4ull << 30;
        e.kvmem_index_placement = "cpu"; e.kvmem_query_conditioned = true; e.kvmem_update_mode = "step";
        auto backend = make_qwen_native_backend(); backend->load(e);
        if (cache_audit) {
            std::vector<std::string> previous;
            unsigned index = 0, divergent = 0;
            const std::vector<bool> samplers = cpu_cache_audit ? std::vector<bool>(8, false)
                : std::vector<bool>{false, false, true, false};
            for (bool gpu : samplers) {
                setenv("QW3_GPU_TARGET_SAMPLING", gpu ? "1" : "0", 1);
                std::vector<std::string> pieces;
                const auto text = backend->generate("", base, [&](const auto &piece) { pieces.push_back(piece); return true; });
                size_t at = 0; while (at < previous.size() && at < pieces.size() && previous[at] == pieces[at]) ++at;
                const bool equal = pieces == previous;
                if (index && !equal) ++divergent;
                std::cout << "cache_audit route=" << mode << " index=" << index << " gpu=" << gpu
                          << " callbacks=" << pieces.size() << " equal_previous=" << equal
                          << " first_difference=" << at << " status=observed\n" << std::flush;
                if (index && !equal) std::cout << "cache_audit_text=" << std::quoted(text) << '\n' << std::flush;
                if (!cpu_cache_audit && index >= 2 && !equal) throw std::runtime_error("warm CPU/GPU cache audit output differs");
                if (pieces.size() != 256) throw std::runtime_error("cache audit token count differs");
                previous = std::move(pieces); ++index;
            }
            if (divergent) throw std::runtime_error("prefix cache changed fixed-seed same-route output");
            std::cout << "cache_stability route=" << mode << " requests=" << index << " status=pass\n" << std::flush;
            continue;
        }
        if (timing) {
            // Establish the query checkpoint before comparing samplers. Cold
            // versus warm cache equivalence is audited separately; both sides
            // of every timing pair must start from the same cache policy.
            setenv("QW3_GPU_TARGET_SAMPLING", "0", 1);
            unsigned tokens = 0;
            backend->generate("", base, [&](const auto &) { ++tokens; return true; });
            if (tokens != 256) throw std::runtime_error("timing priming token count differs");
            std::cout << "timing_cache_prime route=" << mode << " tokens=" << tokens << '\n' << std::flush;
        }
        for (size_t case_index = 0; case_index < cases.size(); ++case_index) {
            const auto &[name, options] = cases[case_index];
            std::vector<std::string> collected[2];
            std::string collected_text[2];
            const bool first_gpu = timing && case_index % 2;
            for (bool gpu : {first_gpu, !first_gpu}) {
                setenv("QW3_GPU_TARGET_SAMPLING", gpu ? "1" : "0", 1);
                std::vector<std::string> pieces;
                const auto start = std::chrono::steady_clock::now();
                const auto text = backend->generate("", options, [&](const auto &piece) { pieces.push_back(piece); return true; });
                const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                if (pieces.size() != unsigned(options.max_tokens)) throw std::runtime_error("wrong generated token count");
                collected[gpu] = pieces; collected_text[gpu] = text;
                std::cout << "sampling_probe route=" << mode << " case=" << name << " gpu=" << gpu
                          << " tokens=" << pieces.size() << " seconds=" << elapsed << " status=pass\n" << std::flush;
            }
            if (collected[0] != collected[1] || collected_text[0] != collected_text[1]) {
                size_t at = 0; while (at < collected[0].size() && collected[0][at] == collected[1][at]) ++at;
                std::cerr << "parity_failure route=" << mode << " case=" << name << " callback=" << at
                          << " cpu=" << std::quoted(collected_text[0]) << " gpu=" << std::quoted(collected_text[1]) << '\n';
                throw std::runtime_error("GPU sampler changed fixed-seed same-route output");
            }
            ++passed;
        }
    }
    std::cout << "gpu_sampling_same_route_pairs=" << passed << " status=pass\n";
    return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
