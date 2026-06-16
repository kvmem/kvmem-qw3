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
    BackendKind backend = BackendKind::QwenNative;
    std::string llama_cli_path = "llama-completion";
    int ctx_size = 262144;
    int threads = 0;
    int gpu_layers = -1;
    int batch_size = 2048;
    bool verbose = false;
    bool native_heavy = true;
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
    //                  (2048 for serving / native generation).
    //    0           : disable chunking entirely (whole-prompt batch). Maximum
    //                  throughput; peak scratch grows linearly with prompt length.
    //   >0           : process prefill in fixed-size chunks of this many tokens.
    int prefill_chunk = -1;
    bool native_mtp_trace = false; // run one optional MTP draft-head diagnostic
    int native_mtp_chain = 1; // one-shot diagnostic default; serve normalizes unset to 0
    bool native_mtp_chain_set = false;
    bool native_mtp_prefix = false; // populate diagnostic MTP prefix KV cache
    bool native_mtp_speculate = false; // run experimental MTP speculative decode
    std::string mtp_policy = "fixed"; // fixed or adaptive
    int mtp_adaptive_min_chain = 0; // 0 = backend default
    int mtp_adaptive_max_chain = 0; // 0 = backend default / mtp_chain
};

struct GenerationOptions {
    int max_tokens = 256;
    float temperature = 0.6f;
    float top_p = 0.95f;
    int top_k = 0; // <=0 disables top-k filtering
    float min_p = 0.0f;
    float presence_penalty = 0.0f;
    float repetition_penalty = 1.0f;
    uint64_t seed = 0;
    bool raw_prompt = false;
    bool ignore_eos = false;
    // Internal serving flag: enqueue this request on the native continuous
    // batching worker when the backend supports it. CLI single-shot generation
    // leaves this false and keeps the original synchronous path.
    bool continuous_batching = false;
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
    uint32_t nextn_predict_layers = 0;
};

struct NativePlanInfo {
    bool supported = false;
    std::string architecture;
    uint32_t n_layers = 0; // main transformer layers executed by qwen-native
    uint32_t n_total_layers = 0; // raw GGUF block_count, including trailing MTP blocks
    uint32_t n_nextn_predict_layers = 0;
    uint32_t n_embd = 0;
    uint32_t n_heads = 0;
    uint32_t n_kv_heads = 0;
    uint32_t n_ctx_train = 0;
    uint64_t n_tensors = 0;
    uint64_t n_bound_tensors = 0;
    uint64_t tensor_bytes = 0;
    uint32_t standard_attention_layers = 0;
    uint32_t recurrent_layers = 0;
    bool mtp_supported = false;
    uint32_t mtp_layer_index = 0;
    uint32_t mtp_bound_tensors = 0;
    std::vector<std::string> missing_tensors;
    std::vector<std::string> mtp_missing_tensors;
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
