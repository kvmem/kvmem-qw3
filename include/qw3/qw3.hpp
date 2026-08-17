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
    // After the configured history threshold, treat every physical prefill
    // chunk as a semantic query: run it provisionally, roll back, select a
    // bounded historical window, then replay the chunk into that window.
    // Request-only and opt-in; the production default remains Pressure.
    SemanticChunk,
};

enum class KvMemLocalCacheMode {
    None,
    Frozen,
    Append,
};

// Request-visible metadata for a named, process-local KVMem checkpoint. The
// checkpoint references the current executor's GPU/CPU/NVMe pool authority and
// is therefore intentionally not portable across server restarts.
struct KvMemLocalCacheInfo {
    bool found = false;
    std::string id;
    std::string status; // ready|evicted|expired|failed
    std::string scope = "local";
    uint64_t version = 0;
    uint32_t position = 0;
    std::string fingerprint;
    int64_t created_at = 0;
    int64_t last_access_at = 0;
    int64_t expires_at = 0; // 0 = no TTL
    uint32_t selected_blocks = 0;
    uint32_t total_blocks = 0;
    uint64_t gpu_bytes = 0;
    uint64_t cpu_bytes = 0;
    uint64_t nvme_bytes = 0;
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
    int kvmem_budget = 131072;    // semantic/decode window tokens per selection
    // Long-prefill pressure window. Zero inherits kvmem_budget, preserving the
    // historical single-budget behavior. A larger value lets history KV be
    // constructed under a wider sink+recent window before the final semantic
    // query contracts the active context back to kvmem_budget.
    int kvmem_prefill_budget = 0;
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
    // Host-side KVMem worker/memory locality. auto follows the active GPU's
    // sysfs PCI locality; off preserves OS scheduling; node:N is an override.
    std::string kvmem_numa_policy = "auto"; // auto|off|node:N
    // Adaptive CPU index scoring. TiledOnePass retains exact per-block
    // statistics in a bounded GPU workspace; TiledTwoPass is the minimum-
    // memory compatibility path.
    std::string kvmem_adaptive_score_mode =
        "auto"; // auto|layer-one-pass|tiled-one-pass|tiled-two-pass
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
    // Orthogonal performance-ablation controls. All default on.
    bool kvmem_opt_stage_out = true;
    bool kvmem_opt_stage_in = true;
    bool kvmem_opt_pack = true;
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
    // Durable context archive (docs/kvmem_context_archive_design.md). When
    // `kvmem_archive_dir` is set, the raw-K and V arenas live inside that
    // directory instead of an ephemeral NVMe cache. "build" writes a new
    // archive; "attach" opens a sealed one read-only, so several processes can
    // run different retrieval or budget settings over the same context.
    std::string kvmem_archive_dir;
    std::string kvmem_archive_mode;  // "" | "build" | "attach"
    // Prefix exposed by a dedicated archive-backed Serve process. Zero uses
    // the complete sealed archive; non-zero values are snapped to a physical
    // block boundary by the backend.
    uint64_t kvmem_archive_tokens = 0;
    // Distance between archived recurrent-state ladder points. Truncating an
    // attached archive to an arbitrary length restores the nearest point at or
    // below it and re-prefills the residual, so this trades archive size
    // against worst-case truncation cost. 0 selects a default.
    uint64_t kvmem_archive_ladder_tokens = 0;
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
    // SemanticChunk controls. Zero start inherits the active prefill budget;
    // zero query tokens scores the complete physical prefill chunk. A non-zero
    // query-token cap takes the suffix of the chunk while replaying the whole
    // chunk, which is useful for scorer-cost ablations.
    uint32_t kvmem_prefill_semantic_start_tokens = 0;
    uint32_t kvmem_prefill_semantic_query_tokens = 0;
    // Optional request-local semantic selection budget in tokens. Zero inherits
    // the engine's --kvmem-budget, which is also the hard upper bound. This
    // does not change the pressure-prefill budget or resize the GPU KV pool.
    uint32_t kvmem_semantic_budget = 0;
    std::string kvmem_session_id;
    // Named process-local checkpoint control. A save captures the completed
    // prefill-only request. A load restores that cache and treats this request's
    // prompt as a continuation fragment. Frozen discards the branch after the
    // request; Append atomically publishes version+1 after a prefill-only append.
    std::string kvmem_cache_save_id;
    std::string kvmem_cache_load_id;
    KvMemLocalCacheMode kvmem_cache_load_mode =
        KvMemLocalCacheMode::None;
    uint64_t kvmem_cache_expected_version = 0;
    bool kvmem_cache_expected_version_set = false;
    uint64_t kvmem_cache_ttl_seconds = 0;
    // Query-conditioned KVMem: half-open token span [begin,end) of the prompt
    // that is the user's question. The executor captures these query rows during
    // prefill and uses them for multi-token block selection. begin==end (default)
    // means no span -> the recency/single-token path runs unchanged.
    uint32_t kvmem_query_begin = 0;
    uint32_t kvmem_query_end = 0;
    // Optional half-open token span [begin,end) that must be checkpointed,
    // pinned, and replayed after semantic selection.  This is deliberately
    // independent of kvmem_query_*: benchmark task instructions and output
    // framing may need to be replayed for answer fidelity while only the raw
    // question contributes Q rows to retrieval scoring.  begin==end preserves
    // the legacy behavior by using kvmem_query_* as the replay span.
    uint32_t kvmem_replay_begin = 0;
    uint32_t kvmem_replay_end = 0;
    // Experimental query-token pruning. When both values are non-zero, the
    // backend greedily decodes `probe_tokens` from the provisional prompt,
    // measures their attention mass over the original query span, rolls the
    // decode back, and scores retrieval with only the top `score_tokens` query
    // rows. Zero keeps the existing full/uniform-sampled scorer unchanged.
    uint32_t kvmem_query_attention_probe_tokens = 0;
    uint32_t kvmem_query_attention_score_tokens = 0;
    // Independent guided-retrieval experiment. The backend appends a private
    // retrieval-planning turn, lets the model reason for at most
    // `thinking_max_tokens`, then captures only the compact standalone query it
    // emits after </think>. The private branch is rolled back before normal
    // query replay/answer generation. Zero keeps this path disabled.
    uint32_t kvmem_query_guided_thinking_max_tokens = 0;
    uint32_t kvmem_query_guided_query_max_tokens = 0;
    // Skip private reasoning and directly rewrite the real request as one
    // retrieval question. This remains independent from normal answer thinking.
    bool kvmem_query_guided_direct = false;
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
    // Production semantic pins derived by protocol adapters. Unlike the
    // diagnostics-only oracle spans below, these remain active for pressure
    // prefill and every later reselection in the request. They still consume
    // the fixed selection budget; the serving layer rejects over-budget pin
    // sets instead of silently dropping a semantic region.
    enum class KvMemPinnedReason : uint8_t {
        SystemControl = 0,
        CurrentQuery = 1,
        LiveToolTrajectory = 2,
        ProjectPolicy = 3,
        ExplicitClientPin = 4,
    };
    struct KvMemPinnedTokenSpan {
        uint32_t begin = 0;
        uint32_t end = 0;
        KvMemPinnedReason reason = KvMemPinnedReason::CurrentQuery;
    };
    std::vector<KvMemPinnedTokenSpan> kvmem_pinned_token_spans;
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
    KvMemLocalCacheInfo kvmem_local_cache_info(const std::string &id);
    bool erase_kvmem_local_cache(const std::string &id);
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
