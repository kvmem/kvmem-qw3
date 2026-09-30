#include "backend.hpp"
#include "qw3/gguf.hpp"
#include "qw3/tokenizer.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/resource.h>

// Manual CUDA integration test, deliberately separate from functional CTest.
// Sends full transcripts through the production backend and rewrites generated
// tails on the following request. No synthetic executor/checkpoint seeding.
int main(int argc, char **argv) {
    try {
        if (argc < 2 || argc > 5)
            throw std::invalid_argument("usage: qw3-kvmem-repository-backend-probe MODEL [ROUNDS=32] [plain|mtp|guided|guided-text|guided-visual] [INITIAL_TOKENS=0]");
        const unsigned rounds = argc >= 3 ? std::stoul(argv[2]) : 32;
        const std::string mode = argc >= 4 ? argv[3] : "guided";
        const bool middecode_rewrite = mode == "middecode-text";
        const bool guided_boundary = std::getenv("QW3_TEST_GUIDED_BOUNDARY") != nullptr;
        if (mode != "plain" && mode != "mtp" && mode != "guided" && mode != "guided-text" && mode != "guided-visual" && !middecode_rewrite)
            throw std::invalid_argument("unknown backend probe mode");
        setenv("QW3_PAGED_KV_PAGE_SIZE", "16", 1);
        setenv("QW3_KV_DTYPE", "fp8", 1);
        setenv("QW3_KVMEM_PREFIX_CACHE_TRACE", "1", 1);
        setenv("QW3_KVMEM_GUIDED_MTP", "1", 1);
        qw3::EngineOptions engine;
        engine.model_path = argv[1];
        engine.ctx_size = 8192;
        engine.prefill_chunk = (mode == "guided-text" || mode == "guided-visual") ? 256 : 128;
        engine.kvmem_enabled = true;
        engine.kvmem_repository_blocks = 256;
        if (const char *mode = std::getenv("QW3_TEST_REPOSITORY_GC"))
            engine.kvmem_repository_gc_mode = mode;
        engine.kvmem_block_tokens = 32;
        engine.kvmem_budget = 1024;
        engine.kvmem_gen_budget = 1024;
        engine.kvmem_sink_blocks = 1;
        engine.kvmem_recent_blocks = 4;
        engine.kvmem_index_placement = "cpu";
        engine.kvmem_cpu_bytes = 1ull << 30;
        engine.kvmem_query_conditioned = true;
        if (guided_boundary) engine.kvmem_update_mode = "step";
        engine.native_mtp_speculate = mode != "plain";
        engine.native_mtp_prefix = mode != "plain";
        engine.native_mtp_chain = mode == "plain" ? 0 : 4;
        engine.native_mtp_chain_set = true;
        if (middecode_rewrite) {
            engine.ctx_size = 32768;
            engine.kvmem_repository_blocks = 1024;
            engine.kvmem_budget = 8192;
            engine.kvmem_gen_budget = 8192;
            engine.kvmem_cpu_bytes = 4ull << 30;
            engine.kvmem_update_mode = "step";
        }
        if (const char *v = std::getenv("QW3_TEST_REPOSITORY_TOKENS")) {
            engine.ctx_size = std::stoi(v);
            engine.prefill_chunk = 2048;
            engine.kvmem_repository_blocks = engine.ctx_size / 32;
            engine.kvmem_budget = engine.kvmem_prefill_budget = 131072;
            engine.kvmem_gen_budget = 131072;
            engine.kvmem_cpu_bytes = 80ull << 30;
            engine.kvmem_update_mode = "step";
        }
        auto backend = qw3::make_qwen_native_backend();
        backend->load(engine);
        std::unique_ptr<qw3::GgufFile> gguf;
        std::unique_ptr<qw3::QwenTokenizer> tokenizer;
        if (mode == "guided-text" || mode == "guided-visual" || middecode_rewrite) {
            gguf = std::make_unique<qw3::GgufFile>(argv[1]);
            tokenizer = std::make_unique<qw3::QwenTokenizer>(*gguf);
        }
        qw3::GenerationOptions request;
        request.temperature = 0;
        request.ignore_eos = true;
        const char *output_setting = std::getenv("QW3_TEST_OUTPUT_TOKENS");
        const int fixed_output = output_setting ? std::stoi(output_setting) : -1;
        std::ofstream events;
        if (const char *path = std::getenv("QW3_TEST_TIMING_EVENTS")) {
            events.open(path);
            if (!events) throw std::runtime_error("cannot create timing events output");
            events << "round,callback,elapsed_ms\n" << std::setprecision(12);
        }
        if (argc == 5) {
            const uint32_t initial = std::stoul(argv[4]);
            for (uint32_t i = 0; i < initial; ++i)
                request.prompt_token_ids_override.push_back(100 + i % 1000);
        }
        std::cout << "round,history,output_limit,elapsed_ms,rss_kib,callbacks,ttft_ms,callback_span_ms\n";
        for (unsigned round = 0; round < rounds; ++round) {
            if (tokenizer) {
                auto append = [&](const std::string &text) {
                    const auto ids = tokenizer->encode(text);
                    request.prompt_token_ids_override.insert(request.prompt_token_ids_override.end(), ids.begin(), ids.end());
                };
                append(round ? "<|im_end|>\n" : "<|im_start|>system\nYou are a debugging assistant. Use the tool log to answer the request.<|im_end|>\n");
                append("<|im_start|>user\nTool log, iteration " + std::to_string(round) +
                    ": read src/client.cpp. Function fetch_record retries network requests. "
                    "The retry limit is three. Failure message: connection reset by peer. "
                    "The timeout is 500 ms. Earlier retries may have used a stale connection. "
                    "The current regression test covers timeout and cancellation handling.\n\n");
                if (mode == "guided-visual") {
                    // Synthetic projected rows exercise real M-RoPE/attention;
                    // this is not a vision encoder or task accuracy benchmark.
                    for (uint32_t i = 0; i < 8; ++i) {
                        qw3::GenerationOptions::InputEmbeddingOverride row;
                        row.token_id = 0x80000001u + request.input_embedding_overrides.size();
                        row.source_token_id = 100; row.cache_identity = row.token_id;
                        const uint32_t pos = request.prompt_token_ids_override.size();
                        row.position = {2*pos, 2*pos+pos%4, 2*pos+pos%2};
                        row.embedding.resize(gguf->find_tensor("token_embd.weight")->dims.at(0));
                        for (size_t j = 0; j < row.embedding.size(); ++j) row.embedding[j] = 0.05f*std::sin(float(i+j));
                        request.prompt_token_ids_override.push_back(row.token_id);
                        request.input_embedding_overrides.push_back(std::move(row));
                    }
                }
                request.kvmem_query_begin = request.prompt_token_ids_override.size();
                append("Find the function name, retry limit and network error from the earlier tool log.");
                if (const char *v = std::getenv("QW3_TEST_QUERY_TOKENS")) {
                    const auto count = std::stoul(v);
                    while (request.prompt_token_ids_override.size() - request.kvmem_query_begin < count)
                        append(" Review the prior evidence and check the retry limit.");
                }
                request.kvmem_query_end = request.prompt_token_ids_override.size();
                append("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
            } else {
                for (unsigned i = 0; i < 128; ++i)
                    request.prompt_token_ids_override.push_back(100 + (round * 128 + i) % 1000);
                request.kvmem_query_begin = request.prompt_token_ids_override.size() - 32;
                request.kvmem_query_end = request.prompt_token_ids_override.size();
            }
            if (mode == "guided-visual") {
                request.input_mrope_positions.resize(request.prompt_token_ids_override.size());
                for (uint32_t i = 0; i < request.input_mrope_positions.size(); ++i)
                    request.input_mrope_positions[i] = {2*i, 2*i+i%4, 2*i+i%2};
            }
            // Warm root-query regression: prefill fits the ordinary GPU pool,
            // but its private retrieval turn needs more than the 55 tokens left.
            // Keep the captured root query unchanged across the first selection.
            if (guided_boundary) {
                if (!tokenizer || round > 1) throw std::runtime_error("boundary probe requires two text/visual requests");
                const uint32_t target = engine.kvmem_budget + engine.kvmem_gen_budget - (round ? 55 : 512);
                request.prompt_token_ids_override.resize(target);
                for (uint32_t i = 0; i < target; ++i) request.prompt_token_ids_override[i] = 100 + i % 1000;
                const auto query = tokenizer->encode("Find the earlier retry limit and network error.");
                request.kvmem_query_begin = 64;
                request.kvmem_query_end = 64 + query.size();
                std::copy(query.begin(), query.end(), request.prompt_token_ids_override.begin() + 64);
                if (mode == "guided-visual") {
                    request.input_embedding_overrides.resize(8);
                    request.input_mrope_positions.resize(target);
                    for (uint32_t i = 0; i < target; ++i)
                        request.input_mrope_positions[i] = {i/2, i/2, i/2};
                    for (uint32_t i = 0; i < 8; ++i) {
                        auto &row = request.input_embedding_overrides[i];
                        row.position = request.input_mrope_positions[128+i];
                        request.prompt_token_ids_override[128+i] = row.token_id;
                    }
                } else {
                    request.input_embedding_overrides.clear();
                    request.input_mrope_positions.clear();
                }
            }
            const auto end = request.prompt_token_ids_override.size();
            const bool select = guided_boundary ? round == 1 :
                !middecode_rewrite && end > uint32_t(engine.kvmem_budget) && round % 4 == 0;
            request.kvmem_prepare_query_only = !select;
            request.kvmem_reselect_mode = select ? qw3::KvMemReselectMode::Auto : qw3::KvMemReselectMode::Off;
            request.kvmem_query_guided_direct = (mode == "guided" || mode == "guided-text" || mode == "guided-visual") && select;
            request.kvmem_query_guided_query_max_tokens = request.kvmem_query_guided_direct ? (tokenizer ? 64 : 8) : 0;
            if (std::getenv("QW3_TEST_QUERY_TOKENS")) {
                request.kvmem_query_guided_direct = false;
                request.kvmem_query_guided_query_max_tokens = 0;
            }
            request.max_tokens = fixed_output >= 0 ? fixed_output : (round % 4 == 0 ? 16 : 0);
            if (guided_boundary) {
                request.kvmem_query_guided_direct = false;
                request.kvmem_query_guided_query_max_tokens = select ? 64 : 0;
                request.max_tokens = select ? 16 : 0;
                request.kvmem_prefill_window_mode = qw3::KvMemPrefillWindowMode::KeepSelected;
            }
            if (middecode_rewrite) {
                request.max_tokens = 192;
                request.kvmem_middecode_trigger_tokens = 13;
                request.kvmem_middecode_steady_trigger_tokens = 1024;
                request.kvmem_middecode_max_refreshes = 1;
                request.kvmem_middecode_query_max_tokens = 64;
            }
            const auto start = std::chrono::steady_clock::now();
            std::vector<double> callback_times;
            const auto generated = backend->generate("", request, [&](const std::string &) {
                callback_times.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count());
                return true;
            });
            const double elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            // Record after completion so filesystem IO is not in the token path.
            if (events) for (size_t i = 0; i < callback_times.size(); ++i)
                events << round << ',' << i << ',' << callback_times[i] << '\n';
            struct rusage resource{};
            getrusage(RUSAGE_SELF, &resource);
            std::cout << round << ',' << end << ',' << request.max_tokens << ','
                << elapsed_ms << ',' << resource.ru_maxrss << ',' << callback_times.size()
                << ',' << (callback_times.empty() ? -1 : callback_times.front())
                << ',' << (callback_times.empty() ? 0 : callback_times.back() - callback_times.front())
                << std::endl;
            if (middecode_rewrite) {
                const auto ids = tokenizer->encode(generated);
                if (ids.size() <= 80) throw std::runtime_error("short output cannot exercise a 68-token tail rewrite");
                request.prompt_token_ids_override.insert(request.prompt_token_ids_override.end(),
                    ids.begin(), ids.end() - 68);
                // The next tool transcript replaces the final 68 generated
                // tokens, as in OSWorld030. LCP must remain after the first
                // aligned checkpoint in the new selection generation.
            }
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "backend probe failed: " << error.what() << '\n';
        return 1;
    }
}
