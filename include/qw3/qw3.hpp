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

enum class KvMemReselectMode {
    Auto,
    Force,
    Off,
};

enum class KvMemPrefillWindowMode {
    Pressure,
    KeepSelected,
};

// Diagnostics-only, one-shot selected-context rebuild. Off is the production
// default. KvOnly refreshes normal-attention K/V in the compact selected
// context while restoring the historical recurrent/conv state at the query
// boundary; KvAndState rebuilds both representations from the selected text.
enum class KvMemInlineRefreshMode {
    Off,
    KvOnly,
    KvAndState,
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
    // Keep a BF16 input embedding table in its mapped host checkpoint and
    // stage only selected rows to CUDA. Opt-in because it trades a small
    // host-gather / PCIe cost for substantially lower device memory use.
    bool cpu_embedding = false;
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

    // ---- Block-sparse KV attention (single-session, opt-in) ---------------
    // Master switch (default OFF). When false the forward path is byte-
    // identical to the pre-block-sparse code. All params below take effect
    // only when this is true.
    bool kvmem_enabled = false;
    int kvmem_block_tokens = 128;        // block granularity (multiple of KV page size)
    int kvmem_budget = 131072;    // max window tokens kept per selection
    int kvmem_gen_budget = 32768; // GPU pool reserve for generated tokens; also caps max_tokens
    int kvmem_interval = 64;      // decode steps between reselections
    // Always-kept prefix/suffix allocation. Negative means derive from the
    // KVMem token budget after block_tokens is known:
    //   sink   = clamp(1% of budget, 1K, 2K)
    //   recent = clamp(8% of budget, 4K, 16K)
    // Explicit token and block forms are mutually exclusive per band. The
    // block fields remain for exact reproduction of older experiments.
    int kvmem_sink_blocks = -1;
    int kvmem_recent_blocks = -1;
    int kvmem_sink_tokens = -1;
    int kvmem_recent_tokens = -1;
    // Selection signal that ranks the middle blocks each reselection:
    // "retrieval" (default, global content similarity, can resurrect dropped
    // blocks), "h2o" (window-local cumulative attention heat, retention only),
    // or "recency" (sink + recent windows only, no learned signal).
    std::string kvmem_method = "retrieval";
    std::string kvmem_select_policy = "topk"; // topk or quota
    std::string kvmem_retrieval_method = "mean-k"; // mean-k, per-token, sub-block-mean-k, key-direction-fixed4, key-direction-adaptive
    // Placement of the all-layer Mean-K/Adaptive retrieval index. "gpu"
    // preserves the resident low-latency path; "cpu" keeps the full FP16 index
    // in pageable host memory and streams bounded tiles through the GPU.
    std::string kvmem_index_placement = "gpu"; // gpu or cpu
    int kvmem_index_staging_mb = 64;           // per GPU/host staging slot
    // Adaptive CPU index scoring: auto uses a one-transfer/one-dot per-layer
    // path when a layer fits the bounded staging cap, otherwise the exact
    // two-pass tiled compatibility path.
    std::string kvmem_adaptive_score_mode =
        "auto"; // auto|layer-one-pass|tiled-two-pass
    int kvmem_subblocks = 4;      // sub-block means per block (sub-block-mean-k only)
    std::string kvmem_subblock_reduce = "max"; // sub-block score reduction: max or sum
    double kvmem_adaptive_gain_1to2 = 0.10;
    double kvmem_adaptive_gain_2to4 = 0.06;
    // Optional logical retrieval grouping. "round" consumes caller-supplied
    // round spans; "message" consumes supplied spans for flattened benchmarks
    // or derives spans from ordinary Chat API messages. Both score at the
    // configured sub-block granularity and materialize complete groups.
    std::string kvmem_semantic_expansion = "none"; // none|round|message
    // Existing MaxSim or globally normalized attention mass divided by the
    // number of scoring slices raised to alpha.
    std::string kvmem_group_score_reduce = "max";
    double kvmem_group_length_alpha = 0.5;
    std::string kvmem_update_mode = "interval"; // interval or step
    // Deprecated cumulative storage/tiering profile. It is consulted only
    // when the CLI explicitly passes --kvmem-optimization-level; otherwise
    // common Opt3 infrastructure backs the independent default-on groups.
    std::string kvmem_optimization_level = "opt_3";
    bool kvmem_optimization_level_explicit = false;
    // Repeatable paper-ablation switch. Empty means all optimizations on.
    // Valid names: proactive-stage-out, hierarchical-reuse,
    // packed-rematerialization, or all.
    std::vector<std::string> kvmem_optimize_off;
    // Archived experimental DeltaNet retrieval. These programmatic fields remain
    // for reproducibility, but the corresponding CLI is disabled and the method
    // is not recommended. See docs/kvmem_deltanet_retrieval_experimental.md.
    int kvmem_deltanet_layers = 0;          // 0 = derive from budget
    std::string kvmem_deltanet_layer_policy = "even"; // even or late
    double kvmem_deltanet_mem_budget_gb = 32.0;
    bool kvmem_deltanet_decay = true;       // apply exp(G_M - G_j)
    int kvmem_deltanet_topk_q = 4;          // TopKMean over query tokens
    int kvmem_deltanet_topk_h = 4;          // TopKMean over heads
    // When true, the serve layer marks the final user message's token span as
    // the "question" and the executor scores blocks by the multi-token mean
    // (mean over question tokens) instead of recency. Default OFF -> behavior is
    // byte-identical to the recency/single-token retrieval path.
    bool kvmem_query_conditioned = false;
    // Re-prefill the query suffix against the just-selected semantic window.
    // Enabled by default for query-conditioned mean-k; the first-pass query is
    // still used for retrieval scoring, while decode consumes the replayed KV.
    bool kvmem_recompute_query = true;
    // Preserve unrotated K in a CPU mirror and keep one active GPU K copy.
    // Cold/periodic refreshes rebuild from raw K; small moves use delta RoPE.
    // Enabled by default; use --no-kvmem-immutable-k for legacy ablations.
    bool kvmem_immutable_source_k = true;
    int kvmem_retrieval_blocks = 0; // 0 = derive from remaining budget
    int kvmem_profile_blocks = 0;   // 0 = derive from remaining budget
    double kvmem_gpu_memory_ratio = 0.50;
    double kvmem_gpu_high_watermark = 0.95;
    double kvmem_gpu_low_watermark = 0.85;
    uint64_t kvmem_cpu_bytes = 0;
    uint64_t kvmem_nvme_bytes = 0;
    std::string kvmem_nvme_dir;
    // Store immutable raw-K in a dedicated SSD backing arena. Disabled by
    // default for compatibility; enable for contexts whose raw-K authority
    // exceeds the CPU tier budget.
    bool kvmem_raw_k_nvme = false;
};

struct GenerationOptions {
    int max_tokens = 256;
    float temperature = 0.6f;
    float top_p = 0.95f;
    int top_k = 20; // Qwen3 recommended default; <=0 disables top-k filtering
    float min_p = 0.0f;
    float presence_penalty = 0.0f;
    float repetition_penalty = 1.0f;
    uint64_t seed = 0;
    bool raw_prompt = false;
    // Exact raw-token prompt override used by the /v1/completions integer-array
    // form. Empty means tokenize the prompt string as usual. This avoids a
    // decode/re-tokenize change at concatenated sparse-block boundaries in
    // controlled representation experiments.
    std::vector<uint32_t> prompt_token_ids_override;
    bool ignore_eos = false;
    // Serving compatibility: if generation is still inside an open <think>
    // block, replace a sampled EOS with the tokenizer's </think> token and
    // continue decoding. EOS remains a normal stop after thinking closes.
    bool recover_thinking_eos = false;
    // Internal serving flag: enqueue this request on the native continuous
    // batching worker when the backend supports it. CLI single-shot generation
    // leaves this false and keeps the original synchronous path.
    bool continuous_batching = false;
    // Thinking budget: cap the number of tokens generated inside the <think>
    // block. 0 disables the cap. When the budget is reached while the block is
    // still open, the engine force-injects a short guidance line and the
    // </think> closing tag so the model proceeds straight to its answer.
    int thinking_budget = 0;
    // Whether the prompt already opened a <think> block (enable_thinking). The
    // budget counter only runs while a think block is open.
    bool thinking_open = false;
    // Generic multi-request KVMem controls. These describe inference behavior;
    // dataset/session parsing remains entirely in the caller.
    KvMemReselectMode kvmem_reselect_mode = KvMemReselectMode::Auto;
    KvMemPrefillWindowMode kvmem_prefill_window_mode =
        KvMemPrefillWindowMode::Pressure;
    std::string kvmem_session_id;
    // Query-conditioned KVMem: half-open token span [begin,end) of the prompt
    // that is the user's question. The executor captures these query rows during
    // prefill and uses them for multi-token block selection. begin==end (default)
    // means no span -> the recency/single-token path runs unchanged.
    uint32_t kvmem_query_begin = 0;
    uint32_t kvmem_query_end = 0;
    // Experimental transcript replay: every span is a user query that arrived
    // after the KVMem working-set + generation reserve was already full. During
    // prefill the backend reselects at each span, replays that query against the
    // new window, then teacher-forces the recorded response. Only the final span
    // is followed by decode. Empty keeps the normal one-shot path unchanged.
    struct KvMemReplayQuerySpan {
        uint32_t begin = 0;
        uint32_t end = 0;
    };
    std::vector<KvMemReplayQuerySpan> kvmem_replay_query_spans;
    // Optional variable-length logical retrieval groups in rendered-prompt
    // token coordinates. Populated by the serving layer in round/message
    // semantic-expansion mode. The model/KV storage remains fixed-block; these
    // spans affect semantic score reduction and selection only.
    struct KvMemRetrievalGroupSpan {
        uint32_t begin = 0;
        uint32_t end = 0;
    };
    std::vector<KvMemRetrievalGroupSpan> kvmem_retrieval_group_spans;
    // Optional transcript-construction boundaries. Each value is the first
    // token of an independent historical session, before block alignment. The
    // session-local canonical-KV experiment uses these boundaries to prevent a
    // block's hidden state from inheriting unrelated preceding sessions.
    std::vector<uint32_t> kvmem_replay_session_starts;
    // Optional diagnostics-only metadata. The context span identifies the
    // flattened benchmark history inside the rendered prompt, while trace_tag
    // provides a stable request/sample key for offline retrieval analysis.
    // These fields never participate in scoring or block selection.
    uint32_t kvmem_context_begin = 0;
    uint32_t kvmem_context_end = 0;
    std::string kvmem_trace_tag;
    // Diagnostics-only oracle control. Each half-open span is expressed in
    // already-rendered prompt-token coordinates and forces every overlapping
    // historical KVMem block into the ordinary fixed-size selection budget.
    // The server accepts this field only when QW3_KVMEM_ENABLE_ORACLE=1.
    // Empty is the production/default path and leaves selection byte-identical.
    struct KvMemOracleTokenSpan {
        uint32_t begin = 0;
        uint32_t end = 0;
    };
    std::vector<KvMemOracleTokenSpan> kvmem_oracle_token_spans;
    // When true, the final-query diagnostic selection contains only sink,
    // oracle-overlapping, and pinned query-tail blocks. Ordinary retrieval
    // candidates do not fill the unused budget.
    bool kvmem_oracle_only = false;
    // Diagnostics-only one-request ablation. After the ordinary long-context
    // prefill and final semantic selection, replay exactly the selected source
    // tokens in compact order and replace the answer-producing cache in memory.
    // The server accepts this only behind QW3_KVMEM_ENABLE_INLINE_REFRESH=1.
    KvMemInlineRefreshMode kvmem_inline_refresh =
        KvMemInlineRefreshMode::Off;
    // ARCHIVED (2026-07-23): the DeltaNet recurrent-state export/import debug
    // interface is intentionally compiled out. See the request-parser note in
    // qw3_server.cpp and KVMI-012 for the measured results and rationale.
#if 0
    std::string kvmem_rebuilt_state_export_key;
    std::string kvmem_rebuilt_state_import_key;
    std::string kvmem_rebuilt_state_capture_key;
    std::string kvmem_rebuilt_state_seed_key;
#endif
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

// Return false to cooperatively stop generation after the current token. This
// lets streaming servers release decode resources promptly after a client
// disconnects or a stop sequence is observed.
using CancellableTokenCallback =
    std::function<bool(const std::string &)>;

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
    // Append a prompt fragment to one persistent native KVMem session. `reset`
    // starts/replaces the session; false continues from the executor's live
    // token/KV/recurrent state without re-tokenizing an earlier prefix.
    void generate_session_stream(const std::string &prompt_fragment,
                                 const GenerationOptions &options,
                                 const TokenCallback &on_text,
                                 bool reset);
    void generate_stream_cancellable(
        const std::string &prompt,
        const GenerationOptions &options,
        const CancellableTokenCallback &on_text);

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
