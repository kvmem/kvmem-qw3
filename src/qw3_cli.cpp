#include "qw3/gguf.hpp"
#include "qw3/qw3.hpp"
#include "server.hpp"

#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace {

void usage(std::ostream &os) {
    os <<
        "Usage: qw3 --model MODEL.gguf -p PROMPT [options]\n"
        "       qw3 serve --model MODEL.gguf [--port 8080] [options]\n"
        "\n"
        "Serve (OpenAI-compatible HTTP API; loads model once, serves forever).\n"
        "  Default is the conservative baseline: one request at a time, FP16 KV,\n"
        "  no global paged KV, no continuous batching, and no MTP.\n"
        "  --port N              Listen port. Default: 8080\n"
        "  --host ADDR           Bind address. Default: 127.0.0.1\n"
        "  --continuous-batching Enable continuous request batching. Also enables\n"
        "                        the required global paged-KV serving pool and\n"
        "                        body-batch executor by default.\n"
        "  --paged-kv            Enable the global paged-KV serving pool.\n"
        "  --body-batch          Enable batched decode body executor.\n"
        "  --max-active N        Max active continuous requests. Default: 2\n"
        "  --max-pending N       Max queued continuous requests. Default: 128\n"
        "  --prefill-burst N     Prefill chunks advanced per scheduler turn.\n"
        "                        Default: max-active\n"
        "  --max-total-tokens N  Total token reservation budget. Default: ctx.\n"
        "                        0 disables this admission budget.\n"
        "  --kv-page-size N      Paged-KV logical/physical page size. Default: 16\n"
        "  --kv-pool-pages N     Global KV pool pages. Default: ceil(ctx/page_size)\n"
        "  --mtp-kv-pool-pages N Global MTP-prefix KV pages. Default: kv-pool-pages\n"
        "  --kv-dtype NAME       KV-cache dtype: fp16 default, or fp8/fp32/q8.\n"
        "  --mtp-chain N         MTP speculative chain length. Default: 0 (off).\n"
        "                        N>0 enables MTP speculation.\n"
        "  --mtp-policy NAME     MTP depth policy: fixed default, or adaptive.\n"
        "  --mtp-adaptive-min-chain N  Adaptive MTP minimum chain. Default: auto.\n"
        "  --mtp-adaptive-max-chain N  Adaptive MTP maximum chain. Default: --mtp-chain.\n"
        "  --mtp-batched-draft   Batch MTP draft projection/FFN/logits.\n"
        "  --mtp-paged-prefix    Use paged MTP prefix KV.\n"
        "  --no-continuous-batching, --no-paged-kv, --no-body-batch,\n"
        "  --no-mtp-batched-draft, --no-mtp-paged-prefix\n"
        "                        Compatibility/debug disable switches.\n"
        "  --enable-thinking     Default chat requests to thinking mode (long CoT).\n"
        "  --thinking-budget N   Default max tokens inside <think> before the\n"
        "                        engine force-closes it. 0 disables (default).\n"
        "  --native-mtp-chain N  Alias for --mtp-chain.\n"
        "  --native-mtp-trace    Diagnostic mode; disables default MTP speculate.\n"
        "  -n N                  Optional service max generated tokens cap.\n"
        "                        Default: use remaining context per request.\n"
        "\n"
        "Runtime:\n"
        "  --backend NAME        qwen-native, mock, or llama-cli. Default: qwen-native\n"
        "  --llama-cli PATH      llama.cpp llama-completion binary. Default: llama-completion\n"
        "  --llama-completion PATH\n"
        "                        Alias for --llama-cli\n"
        "  -m, --model FILE      GGUF model path\n"
        "  -c, --ctx N           Context size. Default: 262144\n"
        "  -t, --threads N       llama.cpp CPU helper threads\n"
        "  -ngl N                GPU layers passed to llama.cpp. Default: -1\n"
        "  -b, --batch N         Batch size passed to llama.cpp. Default: 2048\n"
        "  --native-heavy        Compatibility flag; native generation is enabled by default\n"
        "  --native-kernels NAME cuda. Default: cuda\n"
        "  --native-linear-backend NAME auto, cublas, or custom. Default: auto\n"
        "  --native-mtp-trace    Run one optional MTP draft-head diagnostic\n"
        "  --native-mtp-chain N  Diagnostic MTP draft chain length. Default: 1\n"
        "  --native-mtp-prefix   Populate diagnostic MTP prefix KV before drafts\n"
        "  --native-mtp-speculate Experimental MTP speculative decode\n"
        "  --mtp-policy NAME     MTP depth policy: fixed default, or adaptive.\n"
        "  --native-token-id N   Token id used by qwen-native single-token forward. Default: 0\n"
        "  --prefill-chunk N     Prefill chunk size in tokens (qwen-native).\n"
        "                        0 = no chunking (whole-prompt batch, max throughput,\n"
        "                        peak scratch grows with prompt length).\n"
        "                        N>0 = process prefill in fixed-size chunks.\n"
        "                        Unset = built-in default (2048 for serving).\n"
        "  --no-prefill-chunk    Sugar for --prefill-chunk 0 (max throughput).\n"
        "  --verbose             Keep llama.cpp stderr\n"
        "\n"
        "Prompt:\n"
        "  -p, --prompt TEXT     User prompt\n"
        "  --prompt-file FILE    Read user prompt from file\n"
        "  --system TEXT         System prompt. Default: You are a helpful assistant.\n"
        "  --raw                 Send prompt text without Qwen chat formatting\n"
        "  --think               Do not inject an empty <think> block\n"
        "\n"
        "Sampling:\n"
        "  -n N                  Max generated tokens. Default: 256\n"
        "  --temp F              Temperature. Default: 0.6\n"
        "  --top-p F             Top-p. Default: 0.95\n"
        "  --top-k N             Top-k. Default: 0 (disabled)\n"
        "  --min-p F             Min-p. Default: 0.0\n"
        "  --presence-penalty F  Presence penalty. Default: 0.0\n"
        "  --repetition-penalty F Repetition penalty. Default: 1.0\n"
        "  --seed N              Seed passed to llama.cpp\n"
        "\n"
        "Diagnostics:\n"
        "  --inspect             Print GGUF summary and exit\n"
        "  --native-plan         Build the qwen-native tensor binding and op plan, then exit\n"
        "  --dump-tensors        Print GGUF tensor table and exit\n"
        "  --dump-tokens         Tokenize prompt with the native tokenizer and exit\n"
        "  --dump-logits PATH    Write JSONL per-step top-k logits to PATH for parity diffs\n"
        "  --dump-logits-top-k N Top-K to record (default 16)\n"
        "  -h, --help            Show this help\n";
}

std::string read_file(const std::string &path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open file: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int parse_int(const std::string &s, const std::string &name) {
    size_t pos = 0;
    int v = std::stoi(s, &pos);
    if (pos != s.size()) throw std::runtime_error("invalid integer for " + name + ": " + s);
    return v;
}

float parse_float(const std::string &s, const std::string &name) {
    size_t pos = 0;
    float v = std::stof(s, &pos);
    if (pos != s.size()) throw std::runtime_error("invalid float for " + name + ": " + s);
    return v;
}

uint64_t parse_u64(const std::string &s, const std::string &name) {
    size_t pos = 0;
    uint64_t v = std::stoull(s, &pos);
    if (pos != s.size()) throw std::runtime_error("invalid integer for " + name + ": " + s);
    return v;
}

} // namespace

int main(int argc, char **argv) {
    // Eager-load CUDA modules at process init so subsequent kernel launches
    // hit a stabilized driver registry. Fixes a 15% decode regression that
    // appears once FlashInfer's prefill modules get lazy-loaded mid-run, and
    // also lifts default prefill ~10% by avoiding first-launch load stalls.
    // Must be set before any CUDA call (driver reads it once at cuInit).
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
        setenv("CUDA_MODULE_LOADING", "EAGER", 0);
    }

    qw3::EngineOptions engine;
    qw3::GenerationOptions gen;
    std::string prompt;
    std::string system = "You are a helpful assistant.";
    bool inspect = false;
    bool native_plan = false;
    bool dump_tensors = false;
    bool think = false;

    // `qw3 serve ...` runs the OpenAI-compatible HTTP server instead of a
    // one-shot generate. Detected as the first positional argument.
    bool serve = false;
    qw3::ServerConfig serve_cfg;
    bool kv_dtype_cli_set = false;
    std::string kv_dtype_cli;
    int arg_start = 1;
    if (argc > 1 && std::string(argv[1]) == "serve") {
        serve = true;
        arg_start = 2;
    }

    try {
        for (int i = arg_start; i < argc; ++i) {
            const std::string arg = argv[i];
            auto need = [&](const std::string &name) -> std::string {
                if (++i >= argc) throw std::runtime_error("missing value for " + name);
                return argv[i];
            };

            if (arg == "-h" || arg == "--help") {
                usage(std::cout);
                return 0;
            } else if (arg == "--backend") {
                engine.backend = qw3::parse_backend_kind(need(arg));
            } else if (arg == "--llama-cli") {
                engine.llama_cli_path = need(arg);
            } else if (arg == "--llama-completion") {
                engine.llama_cli_path = need(arg);
            } else if (arg == "-m" || arg == "--model") {
                engine.model_path = need(arg);
            } else if (arg == "-c" || arg == "--ctx") {
                engine.ctx_size = parse_int(need(arg), arg);
            } else if (arg == "-t" || arg == "--threads") {
                engine.threads = parse_int(need(arg), arg);
            } else if (arg == "-ngl") {
                engine.gpu_layers = parse_int(need(arg), arg);
            } else if (arg == "-b" || arg == "--batch") {
                engine.batch_size = parse_int(need(arg), arg);
            } else if (arg == "--native-heavy") {
                engine.native_heavy = true;
            } else if (arg == "--native-kernels") {
                engine.native_kernels = need(arg);
            } else if (arg == "--native-linear-backend") {
                engine.native_linear_backend = need(arg);
            } else if (arg == "--native-mtp-trace") {
                engine.native_mtp_trace = true;
            } else if (arg == "--native-mtp-chain") {
                engine.native_mtp_chain = parse_int(need(arg), arg);
                engine.native_mtp_chain_set = true;
            } else if (arg == "--mtp-chain") {
                engine.native_mtp_chain = parse_int(need(arg), arg);
                engine.native_mtp_chain_set = true;
            } else if (arg == "--mtp-policy") {
                const std::string policy = need(arg);
                if (policy != "fixed" && policy != "adaptive" && policy != "auto") {
                    throw std::runtime_error("invalid --mtp-policy (want fixed|adaptive|auto): " + policy);
                }
                engine.mtp_policy = policy == "auto" ? "adaptive" : policy;
            } else if (arg == "--mtp-adaptive-min-chain") {
                engine.mtp_adaptive_min_chain = parse_int(need(arg), arg);
            } else if (arg == "--mtp-adaptive-max-chain") {
                engine.mtp_adaptive_max_chain = parse_int(need(arg), arg);
            } else if (arg == "--native-mtp-prefix") {
                engine.native_mtp_prefix = true;
            } else if (arg == "--native-mtp-speculate") {
                engine.native_mtp_speculate = true;
            } else if (arg == "--native-token-id") {
                engine.native_token_id = parse_int(need(arg), arg);
            } else if (arg == "--prefill-chunk") {
                engine.prefill_chunk = parse_int(need(arg), arg);
            } else if (arg == "--no-prefill-chunk") {
                engine.prefill_chunk = 0;
            } else if (arg == "--verbose") {
                engine.verbose = true;
            } else if (arg == "-p" || arg == "--prompt") {
                prompt = need(arg);
            } else if (arg == "--prompt-file") {
                prompt = read_file(need(arg));
            } else if (arg == "--system") {
                system = need(arg);
            } else if (arg == "--raw") {
                gen.raw_prompt = true;
            } else if (arg == "--think") {
                think = true;
            } else if (arg == "-n") {
                gen.max_tokens = parse_int(need(arg), arg);
                serve_cfg.default_max_tokens_set = true;
            } else if (arg == "--temp") {
                gen.temperature = parse_float(need(arg), arg);
            } else if (arg == "--top-p") {
                gen.top_p = parse_float(need(arg), arg);
            } else if (arg == "--top-k") {
                gen.top_k = parse_int(need(arg), arg);
            } else if (arg == "--min-p") {
                gen.min_p = parse_float(need(arg), arg);
            } else if (arg == "--presence-penalty") {
                gen.presence_penalty = parse_float(need(arg), arg);
            } else if (arg == "--repetition-penalty") {
                gen.repetition_penalty = parse_float(need(arg), arg);
            } else if (arg == "--seed") {
                gen.seed = parse_u64(need(arg), arg);
            } else if (arg == "--inspect") {
                inspect = true;
            } else if (arg == "--native-plan") {
                native_plan = true;
            } else if (arg == "--dump-tensors") {
                dump_tensors = true;
            } else if (arg == "--dump-tokens") {
                engine.dump_tokens = true;
            } else if (arg == "--dump-logits") {
                engine.dump_logits_path = need(arg);
            } else if (arg == "--dump-logits-top-k") {
                engine.dump_logits_top_k = parse_int(need(arg), arg);
            } else if (arg == "--port") {
                serve_cfg.port = parse_int(need(arg), arg);
            } else if (arg == "--host") {
                serve_cfg.host = need(arg);
            } else if (arg == "--continuous-batching") {
                serve_cfg.continuous_batching = true;
            } else if (arg == "--no-continuous-batching") {
                serve_cfg.continuous_batching = false;
            } else if (arg == "--paged-kv") {
                serve_cfg.paged_kv = true;
                serve_cfg.paged_kv_set = true;
            } else if (arg == "--no-paged-kv") {
                serve_cfg.paged_kv = false;
                serve_cfg.paged_kv_set = true;
            } else if (arg == "--body-batch") {
                serve_cfg.body_batch = true;
                serve_cfg.body_batch_set = true;
            } else if (arg == "--no-body-batch") {
                serve_cfg.body_batch = false;
                serve_cfg.body_batch_set = true;
            } else if (arg == "--max-active") {
                serve_cfg.max_active = parse_int(need(arg), arg);
            } else if (arg == "--max-pending") {
                serve_cfg.max_pending = parse_int(need(arg), arg);
            } else if (arg == "--prefill-burst") {
                serve_cfg.prefill_burst = parse_int(need(arg), arg);
            } else if (arg == "--max-total-tokens") {
                serve_cfg.max_total_tokens = parse_u64(need(arg), arg);
                serve_cfg.max_total_tokens_set = true;
            } else if (arg == "--kv-page-size") {
                serve_cfg.kv_page_size = parse_int(need(arg), arg);
            } else if (arg == "--kv-pool-pages") {
                serve_cfg.kv_pool_pages = parse_int(need(arg), arg);
            } else if (arg == "--mtp-kv-pool-pages") {
                serve_cfg.mtp_kv_pool_pages = parse_int(need(arg), arg);
            } else if (arg == "--mtp-batched-draft") {
                serve_cfg.mtp_batched_draft = true;
                serve_cfg.mtp_batched_draft_set = true;
            } else if (arg == "--no-mtp-batched-draft") {
                serve_cfg.mtp_batched_draft = false;
                serve_cfg.mtp_batched_draft_set = true;
            } else if (arg == "--mtp-paged-prefix") {
                serve_cfg.mtp_paged_prefix = true;
                serve_cfg.mtp_paged_prefix_set = true;
            } else if (arg == "--no-mtp-paged-prefix") {
                serve_cfg.mtp_paged_prefix = false;
                serve_cfg.mtp_paged_prefix_set = true;
            } else if (arg == "--kv-dtype") {
                const std::string dt = need(arg);
                if (dt != "fp16" && dt != "fp32" && dt != "q8" && dt != "fp8") {
                    throw std::runtime_error("invalid --kv-dtype (want fp16|fp32|q8|fp8): " + dt);
                }
                serve_cfg.kv_dtype = dt;
                kv_dtype_cli_set = true;
                kv_dtype_cli = dt;
            } else if (arg == "--enable-thinking") {
                serve_cfg.enable_thinking_default = true;
            } else if (arg == "--thinking-budget") {
                serve_cfg.thinking_budget_default = parse_int(need(arg), arg);
                if (serve_cfg.thinking_budget_default < 0) {
                    throw std::runtime_error("--thinking-budget must be >= 0");
                }
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }

        if (inspect || dump_tensors) {
            const qw3::GgufFile gguf(engine.model_path);
            const qw3::ModelInfo info = gguf.model_info();
            std::cout << "architecture: " << info.architecture << "\n"
                      << "metadata: " << info.metadata_count << "\n"
                      << "tensors: " << info.tensor_count << "\n"
                      << "blocks: " << info.block_count << "\n"
                      << "nextn_predict_layers: " << info.nextn_predict_layers << "\n"
                      << "embedding: " << info.embedding_length << "\n"
                      << "heads: " << info.head_count << "\n"
                      << "kv_heads: " << info.head_count_kv << "\n"
                      << "context: " << info.context_length << "\n";
            if (dump_tensors) {
                std::unordered_map<uint32_t, uint64_t> type_counts;
                std::unordered_map<uint32_t, uint64_t> type_bytes;
                for (const qw3::GgufTensorInfo &t : gguf.tensors()) {
                    type_counts[t.type]++;
                    type_bytes[t.type] += t.bytes;
                    std::cout << "tensor: " << t.name << " type=" << qw3::gguf_tensor_type_name(t.type)
                              << " dims=[";
                    for (size_t i = 0; i < t.dims.size(); ++i) {
                        if (i) std::cout << ",";
                        std::cout << t.dims[i];
                    }
                    std::cout << "] offset=" << t.abs_offset << " bytes=" << t.bytes << "\n";
                }
                std::cout << "tensor_types:\n";
                for (const auto &kv : type_counts) {
                    std::cout << "  " << qw3::gguf_tensor_type_name(kv.first)
                              << " count=" << kv.second
                              << " bytes=" << type_bytes[kv.first] << "\n";
                }
            }
            return 0;
        }

        if (native_plan) {
            const qw3::NativePlanInfo plan = qw3::inspect_native_plan(engine.model_path);
            std::cout << "native backend: " << (plan.supported ? "supported" : "incomplete") << "\n"
                      << "architecture: " << plan.architecture << "\n"
                      << "layers: " << plan.n_layers << "\n"
                      << "total_layers: " << plan.n_total_layers << "\n"
                      << "nextn_predict_layers: " << plan.n_nextn_predict_layers << "\n"
                      << "embedding: " << plan.n_embd << "\n"
                      << "heads: " << plan.n_heads << "\n"
                      << "kv_heads: " << plan.n_kv_heads << "\n"
                      << "context_train: " << plan.n_ctx_train << "\n"
                      << "tensors: " << plan.n_tensors << "\n"
                      << "bound_tensors: " << plan.n_bound_tensors << "\n"
                      << "tensor_bytes: " << plan.tensor_bytes << "\n"
                      << "mtp_supported: " << (plan.mtp_supported ? "yes" : "no") << "\n"
                      << "mtp_layer_index: " << plan.mtp_layer_index << "\n"
                      << "mtp_bound_tensors: " << plan.mtp_bound_tensors << "\n";
            if (!plan.missing_tensors.empty()) {
                std::cout << "missing_tensors:\n";
                for (const std::string &name : plan.missing_tensors) std::cout << "  " << name << "\n";
            }
            if (!plan.mtp_missing_tensors.empty()) {
                std::cout << "mtp_missing_tensors:\n";
                for (const std::string &name : plan.mtp_missing_tensors) std::cout << "  " << name << "\n";
            }
            std::cout << "op_plan:\n";
            for (const std::string &op : plan.op_plan) std::cout << "  " << op << "\n";
            return 0;
        }

        if (serve) {
            engine.backend = qw3::BackendKind::QwenNative;
            engine.native_heavy = true;
            if (engine.native_kernels.empty()) engine.native_kernels = "cuda";
            serve_cfg.default_generation = gen;
            return qw3::run_server(engine, serve_cfg);
        }

        if (kv_dtype_cli_set) {
            setenv("QW3_KV_DTYPE", kv_dtype_cli.c_str(), 1);
        }
        if (engine.mtp_policy != "fixed" && engine.mtp_policy != "adaptive") {
            throw std::runtime_error("invalid --mtp-policy (want fixed|adaptive): " +
                                     engine.mtp_policy);
        }
        if (engine.mtp_adaptive_min_chain < 0) {
            throw std::runtime_error("--mtp-adaptive-min-chain must be >= 0");
        }
        if (engine.mtp_adaptive_max_chain < 0) {
            throw std::runtime_error("--mtp-adaptive-max-chain must be >= 0");
        }
        setenv("QW3_MTP_POLICY", engine.mtp_policy.c_str(), 1);
        if (engine.mtp_adaptive_min_chain > 0) {
            setenv("QW3_MTP_ADAPTIVE_MIN_CHAIN",
                   std::to_string(engine.mtp_adaptive_min_chain).c_str(), 1);
        }
        if (engine.mtp_adaptive_max_chain > 0) {
            setenv("QW3_MTP_ADAPTIVE_MAX_CHAIN",
                   std::to_string(engine.mtp_adaptive_max_chain).c_str(), 1);
        }

        if (prompt.empty()) {
            usage(std::cerr);
            return 2;
        }

        qw3::Engine e(engine);
        const std::string rendered = gen.raw_prompt
            ? prompt
            : qw3::render_qwen3_chat_prompt(system, prompt, think);
        e.generate_stream(rendered, gen, [](const std::string &text) {
            std::cout << text << std::flush;
        });
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "qw3: " << e.what() << "\n";
        return 1;
    }
}
