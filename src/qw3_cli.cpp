#include "qw3/gguf.hpp"
#include "qw3/qw3.hpp"

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
        "\n"
        "Runtime:\n"
        "  --backend NAME        mock, llama-cli, or qwen-native. Default: llama-cli\n"
        "  --llama-cli PATH      llama.cpp llama-completion binary. Default: llama-completion\n"
        "  --llama-completion PATH\n"
        "                        Alias for --llama-cli\n"
        "  -m, --model FILE      GGUF model path\n"
        "  -c, --ctx N           Context size. Default: 32768\n"
        "  -t, --threads N       llama.cpp CPU helper threads\n"
        "  -ngl N                GPU layers passed to llama.cpp. Default: -1\n"
        "  -b, --batch N         Batch size passed to llama.cpp. Default: 2048\n"
        "  --native-heavy        Execute the full native device single-token path\n"
        "  --native-kernels NAME cuda. Default: cuda\n"
        "  --native-linear-backend NAME auto, cublas, or custom. Default: auto\n"
        "  --native-token-id N   Token id used by qwen-native single-token forward. Default: 0\n"
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
    qw3::EngineOptions engine;
    qw3::GenerationOptions gen;
    std::string prompt;
    std::string system = "You are a helpful assistant.";
    bool inspect = false;
    bool native_plan = false;
    bool dump_tensors = false;
    bool think = false;

    try {
        for (int i = 1; i < argc; ++i) {
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
            } else if (arg == "--native-token-id") {
                engine.native_token_id = parse_int(need(arg), arg);
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
            } else if (arg == "--temp") {
                gen.temperature = parse_float(need(arg), arg);
            } else if (arg == "--top-p") {
                gen.top_p = parse_float(need(arg), arg);
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
                      << "embedding: " << plan.n_embd << "\n"
                      << "heads: " << plan.n_heads << "\n"
                      << "kv_heads: " << plan.n_kv_heads << "\n"
                      << "context_train: " << plan.n_ctx_train << "\n"
                      << "tensors: " << plan.n_tensors << "\n"
                      << "bound_tensors: " << plan.n_bound_tensors << "\n"
                      << "tensor_bytes: " << plan.tensor_bytes << "\n";
            if (!plan.missing_tensors.empty()) {
                std::cout << "missing_tensors:\n";
                for (const std::string &name : plan.missing_tensors) std::cout << "  " << name << "\n";
            }
            std::cout << "op_plan:\n";
            for (const std::string &op : plan.op_plan) std::cout << "  " << op << "\n";
            return 0;
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
