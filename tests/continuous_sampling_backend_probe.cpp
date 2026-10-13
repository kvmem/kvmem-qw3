#include "backend.hpp"
#include "env_flags.hpp"
#include "qw3/gguf.hpp"
#include "qw3/tokenizer.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>

// Actual concurrent admission + ragged target execution, CPU/GPU sampler parity.
// This compares the SAME existing CB numerical path. It does not qualify CB
// against single-request BF16 activations or the required task-quality matrix.
int main(int argc, char **argv) try {
    using namespace qw3;
    if (argc < 4 || argc > 6)
        throw std::runtime_error("usage: qw3-continuous-sampling-probe TARGET none|mtp|dflash dense|kvmem [CONCURRENCY=4] [DFLASH_MODEL]");
    const std::string route = argv[2], mode = argv[3];
    const bool matched_kvmem = route == "dflash" || std::getenv("QW3_CB_PROBE_MATCHED_KVMEM");
    const bool serial_controller = std::getenv("QW3_CB_PROBE_SERIAL_CONTROLLER");
    std::ofstream output_transcript;
    if (const char *path = std::getenv("QW3_CB_PROBE_OUTPUT_TRANSCRIPT")) {
        output_transcript.open(path);
        if (!output_transcript) throw std::runtime_error("cannot open response transcript");
    }
    const bool external_cancel = std::getenv("QW3_CB_PROBE_EXTERNAL_CANCEL") != nullptr;
    const bool session_reuse = std::getenv("QW3_CB_PROBE_SESSION_REUSE") != nullptr;
    const bool rotate_session_keys = std::getenv("QW3_CB_PROBE_SESSION_KEY_ROTATION") != nullptr;
    const bool verify_parity = std::getenv("QW3_CB_PROBE_VERIFY_PARITY") != nullptr;
    const bool query_scheduler_parity = std::getenv("QW3_CB_PROBE_QUERY_SCHEDULER_PARITY") != nullptr;
    const bool mixed_query = std::getenv("QW3_CB_PROBE_MIXED_QUERY") != nullptr;
    const bool bounded = std::getenv("QW3_CB_PROBE_BOUNDED") != nullptr;
    if (query_scheduler_parity && std::getenv("QW3_CB_PROBE_PREFILL_SCHEDULER_PARITY"))
        throw std::runtime_error("isolate query and prefill scheduler parity tests");
    const std::string policy = std::getenv("QW3_CB_PROBE_POLICY") ? std::getenv("QW3_CB_PROBE_POLICY") : "query";
    if (policy != "query" && policy != "guided" && policy != "refresh" &&
        policy != "guided-complete" && policy != "refresh-complete" && policy != "refresh-long")
        throw std::runtime_error("unknown controller policy");
    const unsigned concurrency = argc >= 5 ? std::stoul(argv[4]) : 4;
    if ((route != "none" && route != "mtp" && route != "dflash") || (mode != "dense" && mode != "kvmem") ||
        (concurrency != 1 && concurrency != 2 && concurrency != 4 && concurrency != 16))
        throw std::runtime_error("invalid probe arguments: concurrency must be 1, 2, 4, or 16");
    setenv("QW3_KV_DTYPE", "fp8", 1);
    setenv("QW3_CONTINUOUS_BATCHING", "1", 1);
    if (route == "dflash") setenv("QW3_CONTINUOUS_DFLASH", "1", 1);
    setenv("QW3_CONTINUOUS_BATCHING_MAX_ACTIVE", std::to_string(concurrency).c_str(), 1);
    setenv("QW3_CONTINUOUS_BATCHING_ADMISSION_WAIT_US", "50000", 1);
    setenv("QW3_CONTINUOUS_BATCHING_TRACE", "1", 1);
    setenv("QW3_CONTINUOUS_BATCHING_SAMPLING", "1", 1);
    setenv("QW3_CONTINUOUS_BATCHING_KVMEM_BODY", "1", 1);
    setenv("QW3_CONTINUOUS_BATCHING_PREFILL_PACK_RECURRENT_STATE", "1", 1);
    setenv("QW3_CONTINUOUS_BATCHING_RAGGED_PREFILL_MIN_TOKENS", "2", 1);
    if (!std::getenv("QW3_CONTINUOUS_BATCHING_MTP_RAGGED_VERIFY_MIN_TOKENS"))
        setenv("QW3_CONTINUOUS_BATCHING_MTP_RAGGED_VERIFY_MIN_TOKENS", "2", 1);
    setenv("QW3_CONTINUOUS_BATCHING_KV_POOL_PAGES", std::to_string(concurrency * 128).c_str(), 1);
    setenv("QW3_CONTINUOUS_BATCHING_MTP_KV_POOL_PAGES", std::to_string(concurrency * 128).c_str(), 1);
    setenv("QW3_PREFIX_CACHE", "0", 1);
    setenv("QW3_KVMEM_PREFIX_CACHE", session_reuse ? "1" : "0", 1);
    if (session_reuse) setenv("QW3_CONTINUOUS_KVMEM_SESSION_REUSE", "1", 1);
    EngineOptions e;
    e.model_path = argv[1]; e.ctx_size = 2048; e.prefill_chunk = 256;
    // Leave long peers suspended for multiple scheduling rounds after the
    // short request frees a slot, so replenishment exercises held prefill.
    if (std::getenv("QW3_CB_PROBE_PREFILL_ADMISSION_PARITY")) e.prefill_chunk = 32;
    e.native_mtp_chain_set = true; e.native_mtp_chain = route == "mtp" ? 4 : 0;
    e.native_mtp_speculate = route == "mtp";
    if (route == "dflash") {
        e.native_mtp_chain_set = false;
        e.dflash_model_path = argc == 6 ? argv[5] : "models/Qwen3.8-27B-DFlash2-BF16";
    }
    e.kvmem_enabled = mode == "kvmem"; e.kvmem_block_tokens = 32;
    e.kvmem_budget = session_reuse ? 1024 : 256; e.kvmem_gen_budget = policy == "refresh-complete" ? 1024 : 256;
    e.kvmem_sink_blocks = 1; e.kvmem_recent_blocks = 2;
    e.kvmem_cpu_bytes = 256ull << 20; e.kvmem_index_placement = "gpu";
    e.kvmem_query_conditioned = false; e.kvmem_update_mode = "step";
    if (matched_kvmem) {
        e.kvmem_index_placement = "cpu";
        e.kvmem_query_conditioned = true;
    }
    if (bounded) {
        if (mode != "kvmem" || serial_controller)
            throw std::runtime_error("bounded concurrency probe requires the KVMem controller");
        e.ctx_size = 8192; e.kvmem_repository_blocks = 256;
        e.kvmem_cpu_bytes = (uint64_t(concurrency) + 1) << 30;
        e.kvmem_immutable_source_k = true;
        e.kvmem_opt_stage_in = e.kvmem_opt_stage_out = e.kvmem_opt_pack = true;
        e.kvmem_gpu_memory_ratio = .95;
        setenv("QW3_EXPERIMENTAL_BOUNDED_CONTINUOUS", "1", 1);
        setenv("QW3_CONTINUOUS_KVMEM_CONTROLLER", "1", 1);
        setenv("QW3_CONTINUOUS_KVMEM_COOPERATIVE_PREFILL", "1", 1);
        setenv("QW3_KVMEM_PROBE_RESERVE_TOKENS", "2048", 1);
        // This small-budget fixture tests ownership/rollback only. The long
        // benchmark separately requires unchanged K128K/G64K.
        const uint64_t pages = uint64_t(concurrency) * (e.kvmem_budget + e.kvmem_gen_budget + 2048) / 16;
        setenv("QW3_CONTINUOUS_BATCHING_KV_POOL_PAGES", std::to_string(pages).c_str(), 1);
        setenv("QW3_CONTINUOUS_BATCHING_MTP_KV_POOL_PAGES", std::to_string(pages).c_str(), 1);
    }
    GgufFile file(argv[1]); QwenTokenizer tokenizer(file);
    const auto encode = [&](const std::string &s) {
        auto ids = tokenizer.encode(s); return std::vector<uint32_t>(ids.begin(), ids.end());
    };
    auto backend = make_qwen_native_backend(); backend->load(e);
    const unsigned request_count = std::getenv("QW3_CB_PROBE_REPLENISH") ? 2 * concurrency : concurrency;
    std::vector<GenerationOptions> requests(request_count);
    for (unsigned i = 0; i < request_count; ++i) {
        auto &o = requests[i];
        o.continuous_batching = !serial_controller;
        o.max_tokens = 32 + (i % 4) * 8; o.ignore_eos = true;
        o.seed = 101 + i * 37; o.temperature = 1; o.top_k = 20; o.top_p = .95f;
        o.sampling_filter_order = SamplingFilterOrder::TopKFirst;
        const bool private_query_request = !mixed_query || i % concurrency == 0;
        if (mixed_query) o.max_tokens = 96;
        if (matched_kvmem && private_query_request && (policy == "guided" || policy == "guided-complete")) {
            o.kvmem_query_guided_direct = true;
            o.kvmem_query_guided_query_max_tokens = policy == "guided-complete" ? 128 : 16;
        }
        if (matched_kvmem && private_query_request && (policy == "refresh" || policy == "refresh-complete" || policy == "refresh-long")) {
            o.kvmem_middecode_trigger_tokens = 13;
            o.kvmem_middecode_steady_trigger_tokens = 1024;
            o.kvmem_middecode_max_refreshes = 1;
            o.kvmem_middecode_query_max_tokens = policy == "refresh-complete" ? 128 : 16;
            if (policy == "refresh-complete") o.max_tokens = 96;
            if (policy == "refresh-long") {
                o.max_tokens = 320 + (i % 4) * 8;
                o.kvmem_middecode_trigger_tokens = 96;
                o.kvmem_middecode_steady_trigger_tokens = 96;
                o.kvmem_middecode_max_refreshes = 8;
            }
        }
        if (i % 4 == 1) { o.presence_penalty = .7f; o.repetition_penalty = 1.13f; o.min_p = .05f; }
        if (i % 4 == 2) { o.top_k = 0; o.top_p = 1; }
        if (i % 4 == 3) { o.top_k = 180; o.sampling_filter_order = SamplingFilterOrder::Joint; }
        if (verify_parity || std::getenv("QW3_CB_PROBE_GREEDY")) o.temperature = 0;
        auto &prompt = o.prompt_token_ids_override;
        prompt = encode("<|im_start|>user\n");
        const auto line = encode("The retry limit is three. The timeout is 500 milliseconds. Network error: connection reset by peer.\n");
        const bool mixed_prefill = std::getenv("QW3_CB_PROBE_MIXED_PREFILL");
        const size_t length = (mixed_prefill && i % 4 == 0 ? 32 :
                             mode == "kvmem" ? 768 : 32) + 16 * (i % 4);
        while (prompt.size() < length) prompt.insert(prompt.end(), line.begin(), line.end());
        prompt.resize(length);
        const auto query = encode("\nRequest " + std::to_string(i) + ": What is the timeout and retry limit? Explain in detail.");
        if (matched_kvmem) o.kvmem_query_begin = prompt.size();
        prompt.insert(prompt.end(), query.begin(), query.end());
        if (matched_kvmem) o.kvmem_query_end = prompt.size();
        const auto suffix = encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        prompt.insert(prompt.end(), suffix.begin(), suffix.end());
        if (matched_kvmem) o.kvmem_reselect_mode = KvMemReselectMode::Force;
    }
    struct Response { std::string text; std::vector<std::string> pieces; };
    unsigned checks = 0;
    for (bool cancel : {false, true}) {
        std::vector<Response> reference;
        // Repeated CPU control detects nondeterministic scheduling before any
        // GPU-versus-CPU conclusion. Reverse launch order on the final repeat.
        const bool gpu_only = std::getenv("QW3_CB_PROBE_GPU_ONLY");
        const unsigned repeats = std::getenv("QW3_CB_PROBE_CPU_ONLY") || gpu_only ? 2 : 4;
        for (unsigned pass = 0; pass < repeats; ++pass) {
            const bool gpu = gpu_only || pass >= 2;
            setenv("QW3_GPU_TARGET_SAMPLING", gpu ? "1" : "0", 1);
            const bool wide_parity = std::getenv("QW3_CB_PROBE_Q8_WIDE_PARITY");
            const bool wide_swiglu_parity = std::getenv("QW3_CB_PROBE_Q8_WIDE_SWIGLU_PARITY");
            const bool draft_parity = std::getenv("QW3_CB_PROBE_DRAFT_PARITY");
            const bool lt_parity = std::getenv("QW3_CB_PROBE_DRAFT_LT_PARITY");
            const bool prefill_scheduler_parity = std::getenv("QW3_CB_PROBE_PREFILL_SCHEDULER_PARITY");
            const bool prefill_admission_parity = std::getenv("QW3_CB_PROBE_PREFILL_ADMISSION_PARITY");
            if (prefill_admission_parity) {
                if (prefill_scheduler_parity || query_scheduler_parity)
                    throw std::runtime_error("isolate prefill admission from other scheduler parity tests");
                setenv("QW3_CONTINUOUS_KVMEM_COOPERATIVE_PREFILL", "1", 1);
                setenv("QW3_CONTINUOUS_KVMEM_PREFILL_ADMISSION", pass % 2 ? "1" : "0", 1);
            }
            if (prefill_scheduler_parity)
                setenv("QW3_CONTINUOUS_KVMEM_COOPERATIVE_PREFILL", pass % 2 ? "1" : "0", 1);
            if (query_scheduler_parity) {
                setenv("QW3_CONTINUOUS_KVMEM_COOPERATIVE_PREFILL", "1", 1);
                setenv("QW3_CONTINUOUS_KVMEM_COOPERATIVE_QUERY", pass % 2 ? "1" : "0", 1);
            }
            if (std::getenv("QW3_CB_PROBE_MTP_DRAFT_PARITY"))
                setenv("QW3_CONTINUOUS_MTP_BATCHED_DRAFT", pass % 2 ? "1" : "0", 1);
            if (lt_parity)
                setenv("QW3_DFLASH_LT_DRAFT", pass % 2 ? "1" : "0", 1);
            if (draft_parity)
                setenv("QW3_CONTINUOUS_DFLASH_BATCHED_DRAFT", pass % 2 ? "1" : "0", 1);
            if (wide_parity)
                setenv("QW3_Q8_VERIFY_WIDE", pass % 2 ? "1" : "0", 1);
            if (wide_swiglu_parity) {
                if (wide_parity) throw std::runtime_error("isolate wide projection and SwiGLU parity tests");
                setenv("QW3_Q8_VERIFY_WIDE", "1", 1);
                setenv("QW3_Q8_VERIFY_WIDE_SWIGLU", pass % 2 ? "1" : "0", 1);
            }
            if (verify_parity)
                setenv("QW3_CONTINUOUS_BATCHING_RAGGED_PREFILL_EXECUTOR", pass ? "1" : "0", 1);
            std::mutex mutex; std::condition_variable cv;
            unsigned ready = 0; bool start = false;
            std::vector<std::future<Response>> pending(request_count);
            const auto t0 = std::chrono::steady_clock::now();
            for (unsigned j = 0; j < request_count; ++j) {
                const unsigned i = pass % 2 ? request_count - 1 - j : j;
                pending[i] = std::async(std::launch::async, [&, i] {
                    {
                        std::unique_lock<std::mutex> lock(mutex); ++ready; cv.notify_all();
                        cv.wait(lock, [&] { return start; });
                    }
                    Response r;
                    std::atomic<bool> cancelled{false};
                    auto options = requests[i];
                    if (session_reuse)
                        options.kvmem_prefix_cache_key = "probe-" + std::to_string(i) +
                            (rotate_session_keys ? "-" + std::to_string(pass) + "-" + std::to_string(cancel) : "");
                    if (external_cancel) options.should_cancel = [&] { return cancelled.load(); };
                    r.text = backend->generate("", options, [&](const std::string &piece) {
                        r.pieces.push_back(piece);
                        const bool stop = cancel && i % 4 == 1 && r.pieces.size() >= 11;
                        if (external_cancel) { cancelled.store(stop); return true; }
                        return !stop;
                    });
                    return r;
                });
            }
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return ready == request_count; }); start = true; cv.notify_all();
            }
            std::vector<Response> responses;
            for (unsigned i = 0; i < request_count; ++i) {
                auto r = pending[i].get();
                const auto count = cancel && i % 4 == 1 ? 11 : requests[i].max_tokens;
                if (r.pieces.size() != unsigned(count)) throw std::runtime_error("callback count mismatch");
                if (pass && (r.text != reference[i].text || r.pieces != reference[i].pieces)) {
                    size_t at = 0;
                    while (at < r.pieces.size() && at < reference[i].pieces.size() && r.pieces[at] == reference[i].pieces[at]) ++at;
                    std::cerr << "parity_failure pass=" << pass << " gpu=" << gpu << " request=" << i
                              << " cancel=" << cancel << " first_difference=" << at
                              << " expected=" << std::quoted(reference[i].text) << " actual=" << std::quoted(r.text) << '\n';
                    throw std::runtime_error("continuous fixed-seed same-route output changed");
                }
                if (output_transcript.is_open()) {
                    const auto hex = [](const std::string &s) {
                        std::string result; result.reserve(2 * s.size());
                        for (unsigned char c : s) { result += "0123456789abcdef"[c >> 4]; result += "0123456789abcdef"[c & 15]; }
                        return result;
                    };
                    // Stable request order for comparisons across separate
                    // process/backend configurations, including callback cuts.
                    std::ostringstream transcript;
                    transcript << "continuous_output cancel=" << cancel << " pass=" << pass << " request=" << i
                               << " text=" << hex(r.text) << " pieces=";
                    for (const auto &piece : r.pieces) transcript << hex(piece) << ',';
                    output_transcript << transcript.str() << '\n';
                    if (!output_transcript) throw std::runtime_error("cannot write response transcript");
                }
                responses.push_back(std::move(r)); ++checks;
            }
            if (!pass) reference = std::move(responses);
            std::cout << "continuous_sampling route=" << route << " mode=" << mode << " concurrency=" << concurrency
                      << " requests=" << request_count
                      << " policy=" << policy
                      << " matched_kvmem=" << matched_kvmem << " serial_controller=" << serial_controller
                      << " bounded=" << bounded
                      << " session_reuse=" << session_reuse << " rotated_keys=" << rotate_session_keys
                      << " verify_parity=" << verify_parity
                      << " external_cancel=" << external_cancel
                      << " pass=" << pass << " gpu=" << gpu << " cancel=" << cancel
                      << " wide_parity=" << wide_parity << " wide_verify=" << (wide_parity && pass % 2)
                      << " wide_swiglu_parity=" << wide_swiglu_parity << " wide_swiglu=" << (wide_swiglu_parity && pass % 2)
                      << " draft_parity=" << draft_parity << " joint_draft=" << (draft_parity && pass % 2)
                      << " lt_parity=" << lt_parity << " lt_draft=" << (lt_parity && pass % 2)
                      << " prefill_scheduler_parity=" << prefill_scheduler_parity
                      << " prefill_admission_parity=" << prefill_admission_parity
                      << " cooperative_prefill=" << env_flag_enabled("QW3_CONTINUOUS_KVMEM_COOPERATIVE_PREFILL")
                      << " prefill_admission=" << env_flag_enabled("QW3_CONTINUOUS_KVMEM_PREFILL_ADMISSION")
                      << " query_scheduler_parity=" << query_scheduler_parity
                      << " cooperative_query=" << env_flag_enabled("QW3_CONTINUOUS_KVMEM_COOPERATIVE_QUERY")
                      << " mixed_query=" << mixed_query
                      << " seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                      << " status=pass\n" << std::flush;
        }
    }
    std::cout << "continuous_sampling responses=" << checks << " status=pass\n";
    return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
