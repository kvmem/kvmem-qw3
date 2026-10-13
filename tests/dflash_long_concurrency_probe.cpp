// Native request timing, including explicitly grouped resident tool turns.
// Uses frozen real prompt IDs, not HTTP routing.
#include "backend.hpp"
#include "qw3/gguf.hpp"
#include "qw3/tokenizer.hpp"
#include "token_piece_id_map.hpp"
#include "json.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <map>
#include <set>
#include <stdexcept>

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
std::string hex(const std::string &bytes) {
    const char *digits = "0123456789abcdef";
    std::string result; result.reserve(bytes.size() * 2);
    for (unsigned char c : bytes) { result += digits[c >> 4]; result += digits[c & 15]; }
    return result;
}
}

int main(int argc, char **argv) try {
    using namespace qw3;
    require(argc == 5, "usage: PROBE TARGET DRAFT CONFIG_JSON OUTPUT_JSONL");
    std::ifstream input(argv[3]); require(bool(input), "missing config");
    Json config; input >> config;
    const auto route = config.at("route").get<std::string>();
    const unsigned concurrency = config.at("concurrency").get<unsigned>();
    const bool continuous = config.at("continuous").get<bool>();
    const bool resident = config.value("resident_sessions", false);
    const bool independent_streams = config.contains("request_streams");
    const uint32_t repository_blocks = config.value("repository_blocks", 0u);
    require(route == "dflash" || route == "mtp" || route == "none", "invalid route");
    require(concurrency == 1 || concurrency == 2 || concurrency == 4, "invalid concurrency");
    require(continuous || concurrency == 1, "serial reference must have one caller");
    const auto &policy = config.at("policy");
    require(policy.at("k_tokens") == 131072 && policy.at("g_tokens") == 65536,
            "long benchmark must retain K128K/G64K");
    GgufFile file(argv[1]); QwenTokenizer tokenizer(file);
    const test::TokenPieceIdMap piece_ids(tokenizer);
    std::vector<GenerationOptions> requests;
    for (const auto &row : config.at("requests")) {
        GenerationOptions r;
        r.prompt_token_ids_override = row.at("input_ids").get<std::vector<uint32_t>>();
        const auto encoded = tokenizer.encode(row.at("prompt").get<std::string>(), false);
        require(encoded.size() == r.prompt_token_ids_override.size() &&
                std::equal(encoded.begin(), encoded.end(), r.prompt_token_ids_override.begin()),
                "frozen prompt/token IDs differ");
        r.kvmem_query_begin = row.at("query_begin"); r.kvmem_query_end = row.at("query_end");
        require(r.kvmem_query_begin < r.kvmem_query_end && r.kvmem_query_end <= encoded.size(),
                "invalid query span");
        r.kvmem_reselect_mode = KvMemReselectMode::Force;
        r.kvmem_trace_tag = row.at("id").get<std::string>();
        if (resident) {
            r.kvmem_prefix_cache_key = row.at("session_key").get<std::string>();
            require(!r.kvmem_prefix_cache_key.empty(), "resident request needs a session key");
        }
        r.kvmem_middecode_trigger_tokens = row.value("refresh_trigger", 61440u);
        r.kvmem_middecode_steady_trigger_tokens = 61440;
        r.kvmem_middecode_max_refreshes = 2; r.kvmem_middecode_query_max_tokens = 4096;
        r.max_tokens = config.value("max_tokens", 4096);
        require(r.max_tokens > 0 && r.max_tokens <= 4096, "invalid output limit");
        r.ignore_eos = false; r.temperature = 1; r.top_k = 20; r.top_p = .95f;
        r.min_p = 0; r.presence_penalty = 0; r.repetition_penalty = 1;
        r.seed = config.at("seed").get<uint64_t>();
        r.sampling_filter_order = SamplingFilterOrder::TopKFirst;
        r.thinking_open = true; r.thinking_budget = 28672;
        r.continuous_batching = continuous;
        requests.push_back(std::move(r));
    }
    require(!requests.empty(), "empty workload");
    std::vector<std::vector<size_t>> groups;
    if (independent_streams) {
        require(!config.contains("request_groups"), "use either request groups or independent streams");
        groups = config.at("request_streams").get<std::vector<std::vector<size_t>>>();
        require(!groups.empty() && groups.size() <= concurrency, "invalid number of independent streams");
    } else if (config.contains("request_groups")) {
        groups = config.at("request_groups").get<std::vector<std::vector<size_t>>>();
    } else {
        require(!resident, "resident workload needs explicit cold/warm request groups");
        for (size_t base = 0; base < requests.size(); base += concurrency) {
            groups.emplace_back();
            for (size_t i = base; i < std::min(requests.size(), base + concurrency); ++i)
                groups.back().push_back(i);
        }
    }
    std::set<size_t> visited;
    std::map<std::string, size_t> session_stream;
    for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
        const auto &group = groups[group_index];
        require(!group.empty() && (independent_streams || group.size() <= concurrency), "invalid request group width");
        std::set<std::string> keys;
        for (size_t i : group) {
            require(i < requests.size() && visited.insert(i).second, "invalid or repeated request index");
            if (resident) {
                const auto &key = requests[i].kvmem_prefix_cache_key;
                if (independent_streams) {
                    const auto inserted = session_stream.emplace(key, group_index);
                    require(inserted.second || inserted.first->second == group_index,
                            "a session cannot have multiple concurrent stream writers");
                } else require(keys.insert(key).second, "a group cannot write the same session twice");
            }
        }
    }
    require(visited.size() == requests.size(), "schedule must cover every request exactly once");
    setenv("QW3_KV_DTYPE", "fp8", 1);
    setenv("QW3_CONTINUOUS_BATCHING", continuous ? "1" : "0", 1);
    setenv("QW3_CONTINUOUS_DFLASH", "1", 1);
    setenv("QW3_CONTINUOUS_KVMEM_CONTROLLER", "1", 1);
    setenv("QW3_CONTINUOUS_BATCHING_MAX_ACTIVE", std::to_string(concurrency).c_str(), 1);
    setenv("QW3_CONTINUOUS_BATCHING_MAX_TOTAL_TOKENS", "4194304", 1);
    setenv("QW3_CONTINUOUS_BATCHING_ADMISSION_WAIT_US", "50000", 1);
    setenv("QW3_CONTINUOUS_BATCHING_SAMPLING", "1", 1);
    setenv("QW3_CONTINUOUS_BATCHING_KVMEM_BODY", "1", 1);
    setenv("QW3_CONTINUOUS_BATCHING_TRACE", "1", 1);
    setenv("QW3_CONTINUOUS_BATCHING_TIMING", "1", 1);
    setenv("QW3_CONTINUOUS_KVMEM_BATCHED_PREFILL", "0", 1);
    setenv("QW3_PREFIX_CACHE", "0", 1);
    setenv("QW3_KVMEM_PREFIX_CACHE", resident ? "1" : "0", 1);
    setenv("QW3_CONTINUOUS_KVMEM_SESSION_REUSE", resident ? "1" : "0", 1);
    setenv("QW3_PAGED_KV_PAGE_SIZE", "16", 1);
    const uint64_t pages = uint64_t(concurrency) * (131072 + 65536 + 8192) / 16;
    setenv("QW3_CONTINUOUS_BATCHING_KV_POOL_PAGES", std::to_string(pages).c_str(), 1);
    setenv("QW3_CONTINUOUS_BATCHING_MTP_KV_POOL_PAGES", std::to_string(pages).c_str(), 1);
    EngineOptions e;
    e.model_path = argv[1]; e.ctx_size = 1048576; e.prefill_chunk = 2048;
    e.native_mtp_chain_set = true; e.native_mtp_chain = route == "mtp" ? 4 : 0;
    e.native_mtp_speculate = route == "mtp";
    if (route == "dflash") { e.native_mtp_chain_set = false; e.dflash_model_path = argv[2]; }
    e.kvmem_enabled = true; e.kvmem_block_tokens = 32;
    e.kvmem_budget = 131072; e.kvmem_prefill_budget = 131072; e.kvmem_gen_budget = 65536;
    e.kvmem_query_conditioned = true; e.kvmem_retrieval_method = "mean-k";
    e.kvmem_index_placement = "gpu"; e.kvmem_index_staging_mb = 64;
    e.kvmem_update_mode = "step"; e.kvmem_immutable_source_k = true;
    e.kvmem_cpu_bytes = 64ull << 30; e.kvmem_gpu_memory_ratio = .95;
    e.kvmem_opt_stage_in = e.kvmem_opt_stage_out = e.kvmem_opt_pack = true;
    if (repository_blocks) {
        require(continuous && repository_blocks <= uint32_t(INT32_MAX / 32),
                "bounded concurrency needs continuous requests and a valid physical repository");
        e.kvmem_repository_blocks = repository_blocks;
        e.ctx_size = int(repository_blocks * 32);
        e.kvmem_index_placement = "cpu";
        setenv("QW3_EXPERIMENTAL_BOUNDED_CONTINUOUS", "1", 1);
    }
    require(!std::ifstream(argv[4]).good(), "refusing to overwrite results");
    std::ofstream out(argv[4]); require(bool(out), "cannot create results");
    const auto load_start = now_ns();
    auto backend = make_qwen_native_backend(); backend->load(e);
    out << Json{{"type", "metadata"}, {"route", route}, {"concurrency", concurrency},
        {"continuous", continuous}, {"model_load_s", (now_ns()-load_start)/1e9},
        {"k_tokens", e.kvmem_budget}, {"g_tokens", e.kvmem_gen_budget},
        {"gpu_memory_ratio", e.kvmem_gpu_memory_ratio}, {"global_pool_pages", pages},
        {"private_query_reserve_tokens", 8192}, {"prefix_reuse", resident},
        {"independent_streams", independent_streams},
        {"repository_blocks", repository_blocks},
        {"native_callbacks", true}, {"http_service_qualified", false}}.dump() << '\n' << std::flush;
    const auto generate_request = [&](size_t i) {
        std::vector<uint32_t> ids; std::vector<int64_t> times; std::string joined;
        ids.reserve(requests[i].max_tokens); times.reserve(requests[i].max_tokens);
        const auto began = now_ns();
        const auto text = backend->generate("", requests[i], [&](const std::string &piece) {
            const auto at = now_ns();
            ids.push_back(piece_ids.recover(piece)); times.push_back(at); joined += piece;
            return true;
        });
        const auto ended = now_ns();
        require(text == joined, "callbacks do not reconstruct returned bytes");
        require(ids.size() <= unsigned(requests[i].max_tokens), "too many output tokens");
        return Json{{"type", "request"}, {"id", config.at("requests")[i].at("id")},
            {"ordinal", i}, {"input_tokens", requests[i].prompt_token_ids_override.size()},
            {"session_key", requests[i].kvmem_prefix_cache_key},
            {"query_begin", requests[i].kvmem_query_begin}, {"query_end", requests[i].kvmem_query_end},
            {"start_ns", began}, {"end_ns", ended}, {"token_ready_ns", times}, {"token_ids", ids},
            {"response_bytes_hex", hex(text)}, {"output_tokens", ids.size()},
            {"at_token_limit", ids.size() == unsigned(requests[i].max_tokens)}};
    };
    if (independent_streams) {
        // Only the initial arrivals synchronize. Each session immediately
        // submits its next full-history tool turn after its own completion.
        // An unrelated long prefill must not create a workload-wide barrier.
        std::mutex mutex; std::condition_variable cv; size_t ready = 0; bool start = false;
        std::vector<std::future<std::pair<uint64_t, int64_t>>> pending;
        for (size_t stream = 0; stream < groups.size(); ++stream) {
            pending.push_back(std::async(std::launch::async, [&, stream] {
                { std::unique_lock<std::mutex> lk(mutex); ++ready; cv.notify_all();
                  cv.wait(lk, [&] { return start; }); }
                uint64_t tokens = 0; int64_t ended = 0;
                for (size_t turn = 0; turn < groups[stream].size(); ++turn) {
                    auto record = generate_request(groups[stream][turn]);
                    record["stream"] = stream; record["stream_turn"] = turn;
                    tokens += record.at("output_tokens").get<uint64_t>();
                    ended = record.at("end_ns").get<int64_t>();
                    std::lock_guard<std::mutex> lk(mutex);
                    out << record.dump() << '\n' << std::flush;
                    require(bool(out), "cannot write stream response");
                }
                return std::make_pair(tokens, ended);
            }));
        }
        int64_t began;
        { std::unique_lock<std::mutex> lk(mutex); cv.wait(lk, [&] { return ready == groups.size(); });
          began = now_ns(); start = true; cv.notify_all(); }
        uint64_t tokens = 0; int64_t ended = began;
        for (auto &future : pending) {
            const auto result = future.get(); tokens += result.first; ended = std::max(ended, result.second);
        }
        out << Json{{"type", "workload"}, {"request_streams", groups}, {"requests", requests.size()},
            {"start_ns", began}, {"end_ns", ended}, {"output_tokens", tokens},
            {"wall_s", (ended-began)/1e9}}.dump() << '\n' << std::flush;
    } else {
    for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
        const auto &indices = groups[group_index];
        const size_t base = indices.front(), count = indices.size();
        std::mutex mutex; std::condition_variable cv; size_t ready = 0; bool start = false;
        std::vector<std::future<Json>> pending;
        for (size_t offset = 0; offset < count; ++offset) {
            const auto i = indices[offset];
            pending.push_back(std::async(std::launch::async, [&, i] {
                { std::unique_lock<std::mutex> lk(mutex); ++ready; cv.notify_all();
                  cv.wait(lk, [&] { return start; }); }
                return generate_request(i);
            }));
        }
        int64_t began;
        { std::unique_lock<std::mutex> lk(mutex); cv.wait(lk, [&] { return ready == count; });
          began = now_ns(); start = true; cv.notify_all(); }
        uint64_t output_tokens = 0; int64_t last_end = began;
        for (auto &future : pending) {
            auto record = future.get(); output_tokens += record.at("output_tokens").get<uint64_t>();
            last_end = std::max(last_end, record.at("end_ns").get<int64_t>());
            out << record.dump() << '\n' << std::flush;
        }
        out << Json{{"type", "group"}, {"base", base}, {"requests", count},
            {"group_index", group_index}, {"request_indices", indices},
            {"start_ns", began}, {"end_ns", last_end}, {"output_tokens", output_tokens},
            {"wall_s", (last_end-began)/1e9}}.dump() << '\n' << std::flush;
    }
    }
    out << Json{{"type", "completed"}, {"requests", requests.size()}}.dump() << '\n' << std::flush;
    return 0;
} catch (const std::exception &error) { std::cerr << "long concurrency probe: " << error.what() << '\n'; return 1; }
