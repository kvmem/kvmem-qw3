#include "backend.hpp"
#include "qw3/gguf.hpp"
#include "qw3/tokenizer.hpp"
#include "token_piece_id_map.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

// Real-model, same-policy Q8/FP8 production regression. Not registered in CTest:
// requires target/draft checkpoints. Printed timings are diagnostics only.
namespace {
using namespace qw3;
void require(bool ok, const std::string &why) { if (!ok) throw std::runtime_error(why); }
struct Case { std::string name; GenerationOptions request; unsigned cancel_after = 0; };
struct Result {
    std::string text;
    std::vector<std::string> pieces;
    std::vector<uint32_t> token_ids;
};
}
int main(int argc, char **argv) try {
    require(argc >= 3 && argc <= 5, "usage: qw3-dflash-kvmem-lifecycle-probe TARGET DRAFT [HISTORY=4096] [long|guided|semantic|cache]");
    const unsigned history_tokens = argc >= 4 ? std::stoul(argv[3]) : 4096;
    const std::string suite = argc == 5 ? argv[4] : "long";
    require(suite == "long" || suite == "guided" || suite == "semantic" || suite == "cache", "unknown suite");
    require(history_tokens >= 2048 && history_tokens <= 131072, "history must be 2K..128K");
    setenv("QW3_KV_DTYPE", "fp8", 1);
    setenv("QW3_KVMEM_PREFIX_CACHE", "1", 1);
    setenv("QW3_KVMEM_PREFIX_CACHE_TRACE", "1", 1);
    GgufFile file(argv[1]); QwenTokenizer tokenizer(file);
    const qw3::test::TokenPieceIdMap token_ids(tokenizer);
    std::cout << "token_piece_map vocab=" << token_ids.size()
              << " injective=1 mode=inverse_decode_one bytes=raw\n" << std::flush;
    auto encode = [&](const std::string &text) {
        auto ids = tokenizer.encode(text); return std::vector<uint32_t>(ids.begin(), ids.end());
    };
    auto history = encode("<|im_start|>user\nEngineering log.\n");
    const auto line = encode("Function fetch_record retries network requests. The retry limit is three and the timeout is 500 milliseconds. Error: connection reset by peer.\n");
    while (history.size() < history_tokens) history.insert(history.end(), line.begin(), line.end());
    history.resize(history_tokens);
    auto query = [&](std::vector<uint32_t> prefix, const std::string &text, unsigned tokens) {
        GenerationOptions r;
        r.temperature = 0; r.seed = 101; r.max_tokens = tokens; r.ignore_eos = true;
        r.prompt_token_ids_override = std::move(prefix);
        r.kvmem_query_begin = r.prompt_token_ids_override.size();
        auto ids = encode("\n" + text);
        r.prompt_token_ids_override.insert(r.prompt_token_ids_override.end(), ids.begin(), ids.end());
        r.kvmem_query_end = r.prompt_token_ids_override.size();
        ids = encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        r.prompt_token_ids_override.insert(r.prompt_token_ids_override.end(), ids.begin(), ids.end());
        r.kvmem_reselect_mode = KvMemReselectMode::Force;
        return r;
    };
    std::vector<Case> cases;
    GenerationOptions ingest;
    ingest.temperature = 0; ingest.max_tokens = 0; ingest.prompt_token_ids_override = history;
    ingest.kvmem_prepare_query_only = true;
    const std::string question = "What is the function name, retry limit, timeout and network error? Explain how you found the evidence.";
    auto base_query = query(history, question, 64);
    if (suite == "cache") {
        ingest.kvmem_cache_save_id = "dflash-lifecycle";
        cases.push_back({"cache-save", ingest});
        auto branch = query({}, question, 32);
        branch.kvmem_cache_load_id = "dflash-lifecycle";
        branch.kvmem_cache_load_mode = KvMemLocalCacheMode::Frozen;
        branch.kvmem_cache_expected_version_set = true; branch.kvmem_cache_expected_version = 1;
        cases.push_back({"cache-frozen", branch});
        cases.push_back({"cache-frozen-repeat", branch});
        auto append = branch;
        append.max_tokens = 0; append.kvmem_cache_load_mode = KvMemLocalCacheMode::Append;
        cases.push_back({"cache-append", append});
        branch.kvmem_cache_expected_version = 2;
        cases.push_back({"cache-version-2", branch});
    } else if (suite == "semantic") {
        base_query.kvmem_prefill_window_mode = KvMemPrefillWindowMode::SemanticChunk;
        base_query.kvmem_prefill_semantic_start_tokens = 1024;
        base_query.kvmem_prefill_semantic_query_tokens = 32;
        cases.push_back({"semantic-chunk", base_query});
        cases.push_back({"semantic-chunk-repeat", base_query});
    } else {
        cases.push_back({"history-ingest", ingest});
        cases.push_back({"root-query", base_query});
        cases.push_back({"warm-query", base_query});
        if (suite == "guided") {
            auto guided = base_query;
            guided.kvmem_query_guided_direct = true;
            guided.kvmem_query_guided_query_max_tokens = 32;
            cases.push_back({"guided-direct", guided});
            guided.kvmem_query_guided_direct = false;
            guided.kvmem_query_guided_thinking_max_tokens = 16;
            cases.push_back({"guided-private", guided});
            auto attention = base_query;
            attention.kvmem_query_attention_probe_tokens = 16;
            attention.kvmem_query_attention_score_tokens = 8;
            cases.push_back({"attention-probe", attention});
            auto refresh = base_query;
            refresh.max_tokens = 96;
            refresh.kvmem_middecode_trigger_tokens = 13;
            refresh.kvmem_middecode_steady_trigger_tokens = 1024;
            refresh.kvmem_middecode_max_refreshes = 1;
            refresh.kvmem_middecode_query_max_tokens = 16;
            cases.push_back({"middecode-refresh", refresh});
            refresh.kvmem_middecode_query_max_tokens = 128;
            cases.push_back({"middecode-complete-query", refresh});
            cases.push_back({"after-private-branches", base_query});
        } else {
            auto grown = history;
            for (unsigned i = 0; i < 24; ++i) grown.insert(grown.end(), line.begin(), line.end());
            cases.push_back({"history-growth", query(grown, question, 64)});
            cases.push_back({"tail-rewrite", query(grown, "State the timeout and describe the network error.", 64)});
            cases.push_back({"cancel-after-5", base_query, 5});
            cases.push_back({"after-cancel", base_query});
            history[32] = encode(" four").front();
            cases.push_back({"old-prefix-fork", query(history, question, 64)});
        }
    }
    std::vector<Result> reference;
    for (bool use_draft : {false, true}) {
        EngineOptions e; e.model_path = argv[1];
        if (use_draft) e.dflash_model_path = argv[2];
        e.ctx_size = history_tokens + 4096; e.prefill_chunk = 256;
        e.kvmem_enabled = true; e.kvmem_block_tokens = 32;
        e.kvmem_budget = 1024; e.kvmem_gen_budget = 512;
        e.kvmem_sink_blocks = 1; e.kvmem_recent_blocks = 4;
        e.kvmem_cpu_bytes = 4ull << 30; e.kvmem_index_placement = "cpu";
        e.kvmem_query_conditioned = true; e.kvmem_update_mode = "step";
        // These API modes currently use the retained-history store; this is an
        // explicit separate suite, not a claim of bounded-repository support.
        e.kvmem_repository_blocks = (suite == "cache" || suite == "semantic") ? 0 : 256;
        if (e.kvmem_repository_blocks)
            e.ctx_size = e.kvmem_repository_blocks * e.kvmem_block_tokens;
        auto backend = make_qwen_native_backend(); backend->load(e);
        for (size_t i = 0; i < cases.size(); ++i) {
            const auto &c = cases[i]; Result result;
            const auto start = std::chrono::steady_clock::now();
            result.text = backend->generate("", c.request, [&](const std::string &piece) {
                result.token_ids.push_back(token_ids.recover(piece));
                result.pieces.push_back(piece);
                return !c.cancel_after || result.pieces.size() < c.cancel_after;
            });
            require(result.pieces.size() == (c.cancel_after ? c.cancel_after : unsigned(c.request.max_tokens)), c.name + ": callback count");
            std::string joined;
            for (const auto &piece : result.pieces) joined += piece;
            require(joined == result.text, c.name + ": callbacks do not reconstruct returned bytes");
            if (use_draft && (result.text != reference[i].text ||
                             result.token_ids != reference[i].token_ids)) {
                size_t mismatch = 0;
                while (mismatch < result.pieces.size() && mismatch < reference[i].pieces.size() &&
                       result.pieces[mismatch] == reference[i].pieces[mismatch]) ++mismatch;
                std::cerr << "parity_failure case=" << c.name << " first_callback=" << mismatch
                          << " expected=" << std::quoted(reference[i].text) << " actual=" << std::quoted(result.text) << '\n';
                throw std::runtime_error("DFlash lifecycle output/token IDs differ from the same-policy target");
            }
            if (!use_draft) reference.push_back(result);
            if (suite == "cache") {
                auto info = backend->kvmem_local_cache_info("dflash-lifecycle");
                const unsigned expected = i < 3 ? 1 : 2;
                require(info.found && info.status == "ready" && info.version == expected, "cache state/version changed");
                if (i == 2) require(result.text == reference[1].text, "frozen repeat changed");
            }
            std::cout << "route=" << (use_draft ? "dflash" : "plain") << " suite=" << suite
                      << " history=" << history_tokens << " case=" << c.name
                      << " callbacks=" << result.pieces.size() << " wall_s="
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                      << " token_ids=[";
            for (size_t t = 0; t < result.token_ids.size(); ++t) {
                if (t) std::cout << ',';
                std::cout << result.token_ids[t];
            }
            std::cout << "] status=pass\n" << std::flush;
        }
        if (suite == "cache") {
            require(backend->erase_kvmem_local_cache("dflash-lifecycle"), "cache erase failed");
            const auto erased = backend->kvmem_local_cache_info("dflash-lifecycle");
            // The API retains an evicted tombstone for status/version queries.
            require(erased.found && erased.status == "evicted", "cache remained ready after erase");
        }
    }
    return 0;
} catch (const std::exception &e) {
    std::cerr << "DFlash lifecycle probe failed: " << e.what() << '\n'; return 1;
}
