#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace qw3 {

enum class BackendKind {
    Mock,
    LlamaCli,
    QwenNative,
};

struct EngineOptions {
    std::string model_path;
    BackendKind backend = BackendKind::LlamaCli;
    std::string llama_cli_path = "llama-completion";
    int ctx_size = 32768;
    int threads = 0;
    int gpu_layers = -1;
    int batch_size = 2048;
    bool verbose = false;
    bool native_heavy = false;
    int native_token_id = 0;
    std::string native_kernels = "cuda";
    std::string native_linear_backend = "auto";
    // Diagnostics: when non-empty, write a JSONL line per generated step
    // with the prompt tokens, decoded token, and top-k logits.
    std::string dump_logits_path;
    int dump_logits_top_k = 16;
    bool dump_tokens = false; // print tokenized prompt then exit
    // Prefill chunk size override.
    //   -1 (default) : use QW3_PREFILL_CHUNK env var, else built-in default
    //                  (512 — memory parity with llama.cpp).
    //    0           : disable chunking entirely (whole-prompt batch). Maximum
    //                  throughput; peak scratch grows linearly with prompt length.
    //   >0           : process prefill in fixed-size chunks of this many tokens.
    int prefill_chunk = -1;
};

struct GenerationOptions {
    int max_tokens = 256;
    float temperature = 0.6f;
    float top_p = 0.95f;
    uint64_t seed = 0;
    bool raw_prompt = false;
};

struct ModelInfo {
    std::string architecture;
    uint32_t block_count = 0;
    uint32_t embedding_length = 0;
    uint32_t head_count = 0;
    uint32_t head_count_kv = 0;
    uint32_t context_length = 0;
    uint64_t tensor_count = 0;
    uint64_t metadata_count = 0;
};

struct NativePlanInfo {
    bool supported = false;
    std::string architecture;
    uint32_t n_layers = 0;
    uint32_t n_embd = 0;
    uint32_t n_heads = 0;
    uint32_t n_kv_heads = 0;
    uint32_t n_ctx_train = 0;
    uint64_t n_tensors = 0;
    uint64_t n_bound_tensors = 0;
    uint64_t tensor_bytes = 0;
    uint32_t standard_attention_layers = 0;
    uint32_t recurrent_layers = 0;
    std::vector<std::string> missing_tensors;
    std::vector<std::string> op_plan;
};

using TokenCallback = std::function<void(const std::string &)>;

class Engine {
public:
    explicit Engine(EngineOptions options);
    ~Engine();

    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;

    const EngineOptions &options() const;
    ModelInfo inspect_model() const;
    NativePlanInfo native_plan() const;
    std::string generate(const std::string &prompt, const GenerationOptions &options);
    void generate_stream(const std::string &prompt,
                         const GenerationOptions &options,
                         const TokenCallback &on_text);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::string backend_kind_name(BackendKind kind);
BackendKind parse_backend_kind(const std::string &name);
std::string render_qwen3_chat_prompt(const std::string &system,
                                     const std::string &user,
                                     bool enable_thinking);
ModelInfo inspect_gguf(const std::string &path);
NativePlanInfo inspect_native_plan(const std::string &path);

} // namespace qw3
