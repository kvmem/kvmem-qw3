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
#include <time.h>

// Frozen C1 observer: retained cold 4K, warm 32K, bounded cold 128K.
// Callback timestamps measure native token availability, including bursts from
// one verification batch. They are not individual GPU kernel completion times.
namespace {
using Json = nlohmann::json;
void require(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
int64_t monotonic_ns() {
    timespec t{};
    require(clock_gettime(CLOCK_MONOTONIC, &t) == 0, "cannot read CLOCK_MONOTONIC");
    return int64_t(t.tv_sec) * 1000000000ll + t.tv_nsec;
}
std::string hex(const std::string &s) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out; out.reserve(s.size() * 2);
    for (unsigned char c : s) { out += digits[c >> 4]; out += digits[c & 15]; }
    return out;
}
}
int main(int argc, char **argv) try {
    using namespace qw3;
    require(argc == 5, "usage: C1_PROBE TARGET DRAFT INPUT_JSON OUTPUT_JSONL");
    std::ifstream f(argv[3]); require(bool(f), "missing input JSON");
    Json config; f >> config;
    const auto &cell = config.at("cell");
    const auto &policy = config.at("policy");
    const bool warm = cell.at("id") == "sparse_32k_w256_c1_warm";
    const bool cold = cell.at("id") == "full_4k_c1_cold";
    const bool bounded = cell.at("id") == "bounded_128k_w8k_c1_cold";
    require(cell.at("client_concurrency") == 1 &&
        ((cold && cell.at("repository_blocks") == 0 && cell.at("state") == "cold" &&
          cell.at("working_tokens") == 8192 && cell.at("ctx_size") == 10240) ||
         (warm && cell.at("repository_blocks") == 0 && cell.at("state") == "warm" &&
          cell.at("working_tokens") == 256 && cell.at("ctx_size") == 36864) ||
         (bounded && cell.at("repository_blocks") == 512 && cell.at("repository_gc_mode") == "delete" &&
          cell.at("state") == "cold" && cell.at("working_tokens") == 8192 &&
          cell.at("ctx_size") == 16384 && cell.at("history_tokens") == 131072)),
        "worker only implements the three frozen single-request cells");
    const auto route = config.at("route").get<std::string>();
    require(route == "none" || route == "mtp" || route == "dflash", "unknown route");
    const auto mode = config.at("generation").get<std::string>();
    require(mode == "greedy" || mode == "sampling", "unknown generation mode");
    setenv("QW3_KV_DTYPE", "fp8", 1);
    setenv("QW3_CONTINUOUS_BATCHING", "0", 1);
    setenv("QW3_PREFIX_CACHE", "0", 1);
    setenv("QW3_KVMEM_PREFIX_CACHE", warm ? "1" : "0", 1);
    setenv("QW3_KVMEM_PREFIX_CACHE_TRACE", "1", 1);
    if (warm || bounded) setenv("QW3_KVMEM_PERF_TRACE", "1", 1);
    setenv("QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS", "1", 1);
    GgufFile file(argv[1]); QwenTokenizer tokenizer(file);
    const test::TokenPieceIdMap id_map(tokenizer);
    // Complete host validation before model load or any request timing.
    for (const auto &row : config.at("requests")) {
        const auto ids = row.at("input_ids").get<std::vector<uint32_t>>();
        const auto actual = tokenizer.encode(row.at("prompt").get<std::string>(), false);
        require(ids.size() == actual.size() && std::equal(ids.begin(), ids.end(), actual.begin()),
                "frozen prompt/token IDs differ");
        require(row.at("query_begin").get<size_t>() < row.at("query_end").get<size_t>() &&
                row.at("query_end").get<size_t>() <= ids.size(), "invalid frozen query span");
        require(row.at("history_tokens") == cell.at("history_tokens"), "wrong frozen history length");
        if (warm) require(row.at("history_ingest_ids_count") == cell.at("history_tokens") &&
            row.at("history_ingest_ids_count").get<size_t>() <= row.at("query_begin").get<size_t>(),
            "history ingestion must stop before the frozen query");
    }
    EngineOptions e; e.model_path = argv[1];
    e.ctx_size = cell.at("ctx_size"); e.prefill_chunk = policy.at("prefill_chunk");
    e.native_mtp_chain_set = true; e.native_mtp_chain = route == "mtp" ? policy.at("native_mtp_chain").get<int>() : 0;
    e.native_mtp_speculate = route == "mtp";
    if (route == "dflash") { e.native_mtp_chain_set = false; e.dflash_model_path = argv[2]; }
    e.kvmem_enabled = true; e.kvmem_block_tokens = policy.at("block_tokens");
    e.kvmem_repository_blocks = cell.at("repository_blocks");
    e.kvmem_repository_gc_mode = cell.at("repository_gc_mode");
    e.kvmem_budget = cell.at("working_tokens");
    e.kvmem_gen_budget = policy.at("generation_reserve_tokens");
    e.kvmem_prefill_budget = policy.at("prefill_budget");
    e.kvmem_sink_blocks = policy.at("sink_blocks"); e.kvmem_recent_blocks = policy.at("recent_blocks");
    e.kvmem_cpu_bytes = cell.at("cpu_tier_bytes_per_request");
    e.kvmem_index_placement = policy.at("index_placement");
    e.kvmem_retrieval_method = "mean-k"; e.kvmem_query_conditioned = true;
    e.kvmem_update_mode = policy.at("update_mode"); e.kvmem_interval = policy.at("interval");
    e.kvmem_immutable_source_k = true;
    std::ifstream exists(argv[4]); require(!exists.good(), "refusing to overwrite output");
    std::ofstream out(argv[4]); require(bool(out), "cannot open output");
    const auto epoch = monotonic_ns();
    auto now = [&]() { return double(monotonic_ns() - epoch) / 1e9; };
    auto backend = make_qwen_native_backend(); backend->load(e);
    const double load_s = now();
    out << Json{{"type", "metadata"}, {"route", route}, {"generation", mode},
        {"cell", cell.at("id")}, {"model_load_s", load_s},
        {"epoch_monotonic_ns", epoch},
        {"token_piece_map_size", id_map.size()}, {"token_id_observation", "inverse_injective_native_decode_one_raw_bytes"},
        {"time_origin", "steady_clock_before_model_load"}, {"performance_qualified", false},
        {"limitations", "No allocator high-water/per-subsystem resource counters; one process; diagnostic timings only"}}.dump() << '\n';
    for (const auto &row : config.at("requests")) {
        GenerationOptions r;
        const bool warmup = row.at("role") == "warmup";
        r.max_tokens = warmup ? 64 : cell.at("max_new_tokens").get<int>();
        const auto &s = policy.at("sampling");
        r.temperature = mode == "greedy" ? 0.f : s.at("temperature").get<float>();
        r.top_k = s.at("top_k"); r.top_p = s.at("top_p"); r.min_p = s.at("min_p");
        r.presence_penalty = s.at("presence_penalty"); r.repetition_penalty = s.at("repetition_penalty");
        r.ignore_eos = false; r.sampling_filter_order = SamplingFilterOrder::TopKFirst;
        r.seed = 104729 + 37 * row.at("ordinal").get<uint64_t>();
        r.prompt_token_ids_override = row.at("input_ids").get<std::vector<uint32_t>>();
        r.kvmem_query_begin = row.at("query_begin"); r.kvmem_query_end = row.at("query_end");
        r.kvmem_reselect_mode = KvMemReselectMode::Force;
        if (config.value("retrieval_dump", false)) {
            r.kvmem_trace_tag = row.at("id").get<std::string>();
            r.kvmem_context_end = row.at("history_ingest_ids_count");
        }
        double ingest_start = 0, ingest_end = 0;
        if (warm) {
            GenerationOptions ingest;
            ingest.max_tokens = 0; ingest.temperature = 0;
            ingest.kvmem_prepare_query_only = true;
            const auto count = row.at("history_ingest_ids_count").get<size_t>();
            ingest.prompt_token_ids_override.assign(r.prompt_token_ids_override.begin(),
                r.prompt_token_ids_override.begin() + count);
            std::cerr << "c1_ingest_begin id=" << row.at("id") << " history_tokens=" << count << '\n';
            ingest_start = now();
            const auto text = backend->generate("", ingest, [](const std::string &) -> bool {
                throw std::runtime_error("history-only ingestion emitted a token");
            });
            ingest_end = now();
            require(text.empty(), "history-only ingestion returned an answer");
            std::cerr << "c1_ingest_end id=" << row.at("id") << " wall_s=" << ingest_end-ingest_start << '\n';
        }
        std::vector<uint32_t> ids; std::vector<double> ready; std::string joined;
        ids.reserve(r.max_tokens); ready.reserve(r.max_tokens);
        std::cerr << "c1_request_begin id=" << row.at("id") << " route=" << route << '\n';
        const double start = now();
        const auto response = backend->generate("", r, [&](const std::string &piece) {
            const double at = now();
            ids.push_back(id_map.recover(piece)); ready.push_back(at); joined += piece;
            return true;
        });
        const double end = now();
        require(joined == response, "per-token callbacks do not reconstruct returned bytes");
        require(ids.size() <= unsigned(r.max_tokens), "output exceeds token limit");
        Json result{{"type", "request"}, {"id", row.at("id")}, {"ordinal", row.at("ordinal")},
            {"role", row.at("role")}, {"route", route}, {"seed", r.seed},
            {"input_ids_sha256", row.at("input_ids_sha256")}, {"input_tokens", r.prompt_token_ids_override.size()},
            {"query_begin", r.kvmem_query_begin}, {"query_end", r.kvmem_query_end},
            {"start_s", start}, {"end_s", end}, {"wall_s", end-start},
            {"token_ids", ids}, {"token_ready_s", ready}, {"response_bytes_hex", hex(response)},
            {"generated_tokens", ids.size()}, {"at_token_limit", ids.size() == unsigned(r.max_tokens)}};
        if (warm) {
            result["history_ingest_start_s"] = ingest_start;
            result["history_ingest_end_s"] = ingest_end;
            result["history_ingest_s"] = ingest_end-ingest_start;
            result["history_ingest_tokens"] = row.at("history_ingest_ids_count");
            result["ingest_inclusive_wall_s"] = end-ingest_start;
        }
        result["ttft_s"] = ready.empty() ? Json(nullptr) : Json(ready.front()-start);
        result["decode_tokens_per_s"] = ready.size() < 2 || ready.back() == ready.front()
            ? Json(nullptr) : Json((ready.size()-1)/(ready.back()-ready.front()));
        out << result.dump() << '\n' << std::flush;
        std::cerr << "c1_request_end id=" << row.at("id") << " tokens=" << ids.size()
                  << " wall_s=" << end-start << '\n';
    }
    return 0;
} catch (const std::exception &err) {
    std::cerr << "C1 probe failed: " << err.what() << '\n'; return 1;
}
