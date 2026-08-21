#include "qw3/gguf.hpp"
#include "qw3/kvmem_archive.hpp"
#include "qw3/qw3.hpp"
#include "qw3/tokenizer.hpp"
#include "kvmem_archive_cli.hpp"
#include "kvmem_session.hpp"
#include "server.hpp"
#include "json.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void usage(std::ostream &os) {
    os <<
        "Usage: qw3 --model MODEL -p PROMPT [options]\n"
        "       qw3 serve --model MODEL [--port 8080] [options]\n"
        "       qw3 kvmem-session --model MODEL [--session-ladder L] [options]\n"
        "       qw3 archive build --model MODEL --kvmem-archive DIR --ctx N\n"
        "                         [--archive-input FILE] [--archive-tokens N]\n"
        "                         [--archive-token-input FILE]\n"
        "                         [--archive-ladder-tokens N]\n"
        "                         [--archive-pad-final-chunk] [options]\n"
        "                         [--archive-prefill-window pressure|semantic_chunk]\n"
        "                         [--archive-prefill-semantic-start-tokens N]\n"
        "                         [--archive-prefill-semantic-query-tokens N]\n"
        "       qw3 archive query --model MODEL --kvmem-archive DIR\n"
        "                         [--archive-tokens N] --archive-question Q\n"
        "                         [--archive-questions-file FILE]\n"
        "                         [--archive-questions-json FILE]\n"
        "                         [--archive-question-format raw|qwen-chat|qwen-chat-no-thinking|qwen-user-continuation]\n"
        "                         [--archive-query-token-selector none|attention-probe|guided-query]\n"
        "                         [--archive-query-probe-tokens N]\n"
        "                         [--archive-query-score-tokens N]\n"
        "                         [--archive-results-file FILE]\n"
        "                         [options]\n"
        "       qw3 archive info  --kvmem-archive DIR\n"
        "       qw3 tokenize --model MODEL --prompt-file FILE [--token-output FILE]\n"
        "\n"
        "Serve (OpenAI- and Anthropic-compatible HTTP APIs; loads model once).\n"
        "  Default is the conservative baseline: one request at a time, FP16 KV,\n"
        "  no global paged KV, no continuous batching, and no MTP.\n"
        "  --port N              Listen port. Default: 8080\n"
        "  --host ADDR           Bind address. Default: 127.0.0.1\n"
        "  --kvmem-archive DIR  Start a dedicated frozen archive server; each\n"
        "                        request prompt is the new query against the\n"
        "                        attached prefix. Use --archive-tokens N to\n"
        "                        expose a block-aligned prefix (default: all).\n"
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
        "  --prefix-cache        Enable lossless prefix KV caching: reuse the KV\n"
        "                        of a shared prompt prefix across requests (re-ask\n"
        "                        / multi-turn append). Continuous-batching path;\n"
        "                        caches main and MTP draft state. Default: off.\n"
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
        "  -m, --model PATH      GGUF file or HF safetensors model directory\n"
        "  -c, --ctx N           Context size. Default: 262144\n"
        "  -t, --threads N       llama.cpp CPU helper threads\n"
        "  -ngl N                GPU layers passed to llama.cpp. Default: -1\n"
        "  -b, --batch N         Batch size passed to llama.cpp. Default: 2048\n"
        "  --native-kernels NAME cuda. Default: cuda\n"
        "  --native-linear-backend NAME auto, cublas, or custom. Default: auto\n"
        "  --cpu-embedding       Keep a BF16 input embedding table on CPU and\n"
        "                        transfer only selected rows. Requires a separate\n"
        "                        LM head. Default: off.\n"
        "  --native-mtp-trace    Run one optional MTP draft-head diagnostic\n"
        "  --native-mtp-chain N  Diagnostic MTP draft chain length. Default: 1\n"
        "  --native-mtp-prefix   Populate diagnostic MTP prefix KV before drafts\n"
        "  --native-mtp-speculate Experimental MTP speculative decode\n"
        "  --mtp-policy NAME     MTP depth policy: fixed default, or adaptive.\n"
        "  --prefill-chunk N     Prefill chunk size in tokens (qwen-native).\n"
        "                        0 = no chunking (whole-prompt batch, max throughput,\n"
        "                        peak scratch grows with prompt length).\n"
        "                        N>0 = process prefill in fixed-size chunks.\n"
        "                        Unset = built-in default (2048 for serving).\n"
        "  --no-prefill-chunk    Sugar for --prefill-chunk 0 (max throughput).\n"
        "  --kvmem     Enable kvmem block-sparse KV attention (qwen-native,\n"
        "                        single-session). Default OFF. When off the\n"
        "                        forward path is byte-identical to the default.\n"
        "  --kvmem-block-tokens N   Block granularity in tokens (multiple of KV page\n"
        "                        size). Default: 128.\n"
        "  --kvmem-budget N  Maximum semantic/decode window tokens per request.\n"
        "                        Requests may narrow it with kvmem_semantic_budget.\n"
        "                        Default: 131072.\n"
        "  --kvmem-prefill-budget N  Pressure window used while ingesting long\n"
        "                        history. Must be >= --kvmem-budget and defaults\n"
        "                        to the same value when omitted.\n"
        "  --kvmem-interval N  Decode steps between reselections. Default: 64.\n"
        "  --kvmem-sink-tokens N    Always-kept prefix tokens (rounded to blocks).\n"
        "  --kvmem-recent-tokens N  Always-kept suffix tokens (rounded to blocks).\n"
        "                        Defaults derive from --kvmem-budget: sink=clamp(1%,\n"
        "                        1K,2K), recent=clamp(8%,4K,16K).\n"
        "  --kvmem-sink-blocks N    Compatibility override in physical blocks.\n"
        "  --kvmem-recent-blocks N  Compatibility override in physical blocks.\n"
        "                        Token and block forms are mutually exclusive.\n"
        "  --kvmem-method M  Block selection signal: retrieval|h2o|recency.\n"
        "                        Default: retrieval.\n"
        "  --kvmem-select-policy M  Selection policy: topk|quota. Default: topk.\n"
        "  --kvmem-retrieval-method M  Query-conditioned scorer: mean-k|per-token|\n"
        "                        sub-block-mean-k|key-direction-fixed4|\n"
        "                        key-direction-adaptive.\n"
        "                        Fixed4 clusters every 32-token slice inside a\n"
        "                        retrieval block into four direction prototypes.\n"
        "                        Adaptive retains 1, 2, or 4 packed prototypes\n"
        "                        according to normalized residual gain.\n"
        "  --kvmem-adaptive-gain-1to2 F  Minimum fractional residual reduction\n"
        "                        required to retain two prototypes. Default: 0.10.\n"
        "  --kvmem-adaptive-gain-2to4 F  Minimum fractional residual reduction\n"
        "                        required to retain four prototypes. Default: 0.06.\n"
        "                        Default: mean-k\n"
        "                        (needs --kvmem-query-conditioned).\n"
        "  --kvmem-index-placement P  Retrieval index placement: gpu|cpu.\n"
        "                        GPU keeps the complete Mean-K/Adaptive index;\n"
        "                        CPU streams exact tiles through bounded GPU\n"
        "                        staging. Default: gpu.\n"
        "  --kvmem-index-staging-mb N  Per-slot CPU/GPU index staging size.\n"
        "                        Two slots are allocated. Default: 64 MiB.\n"
        "  --kvmem-numa-policy P  Host KVMem locality: auto|off|node:N.\n"
        "                        Auto follows the active GPU PCI locality;\n"
        "                        unsupported systems fall back to off.\n"
        "  --kvmem-adaptive-score-mode M  CPU Adaptive scorer:\n"
        "                        auto|layer-one-pass|tiled-one-pass|tiled-two-pass.\n"
        "                        Auto prefers one H2D transfer and one dot pass\n"
        "                        per layer, with exact tiled fallback.\n"
        "  --kvmem-semantic-expansion M  Complete-group materialization:\n"
        "                        none|round|message. Message spans are derived from\n"
        "                        Chat messages or supplied by flattened benchmarks.\n"
        "  --kvmem-group-score-reduce M  Logical-group score: max or\n"
        "                        length-normalized-mass. Default: max.\n"
        "  --kvmem-group-length-alpha F  Length exponent for normalized mass.\n"
        "                        0=raw mass, 1=mean density. Default: 0.5.\n"
        "  --kvmem-round-retrieval  Compatibility alias for\n"
        "                        --kvmem-semantic-expansion round.\n"
        "  --kvmem-update-mode M  Reselect cadence: interval|step. Default: interval.\n"
        "  --kvmem-opt-stage-out on|off  Proactive lower-tier writeback and\n"
        "                        clean backing. Default: on.\n"
        "  --kvmem-opt-stage-in on|off  GPU selection-delta reuse, incremental\n"
        "                        rematerialization, and heat-aware CPU policy.\n"
        "                        Default: on.\n"
        "  --kvmem-opt-pack on|off  Cross-block D2H/H2D packing, persistent CPU\n"
        "                        workers, coalesced SSD I/O, and batched GPU\n"
        "                        scatter/re-RoPE. Default: on.\n"
        "  --kvmem-query-conditioned  Score blocks by the multi-token mean against the\n"
        "                        final user message (the question) instead of recency.\n"
        "                        Requires the serve layer to mark the query span.\n"
        "  --no-kvmem-recompute-query  Do not replay the query after semantic\n"
        "                        reselection. Default: replay is enabled.\n"
        "  --kvmem-immutable-k  Keep unrotated K in a CPU mirror and one active GPU\n"
        "                        K copy; periodically rebuild to bound re-RoPE drift.\n"
        "                        Default: enabled.\n"
        "  --no-kvmem-immutable-k  Use the legacy in-place K re-RoPE path.\n"
        "  --kvmem-retrieval-blocks N  Quota policy retrieval blocks (0 = derive).\n"
        "  --kvmem-profile-blocks N    Quota policy profile blocks (0 = derive).\n"
        "  --kvmem-gpu-memory-ratio F  GPU memory fraction for KVMem KV cap.\n"
        "                        Default: 0.50.\n"
        "  --kvmem-gpu-low-watermark F  Evict target for future tiering. Default: 0.85.\n"
        "  --kvmem-cpu-gb F      CPU tier budget in GiB for offloaded KV blocks.\n"
        "                        0 disables runtime page release. Default: 0.\n"
        "  --kvmem-cpu-bytes N   CPU tier budget in bytes (legacy alias).\n"
        "  --kvmem-nvme-dir DIR  Directory for KVMem NVMe backing file.\n"
        "  --kvmem-nvme-gb F     NVMe tier budget in GiB. Requires --kvmem-nvme-dir.\n"
        "  --kvmem-nvme-bytes N  NVMe tier budget in bytes (legacy alias).\n"
        "  --kvmem-raw-k-nvme    Reserve NVMe capacity for immutable raw-K and use\n"
        "                        CPU raw-K chunks as a bounded cache. Default: off.\n"
        "  --no-kvmem-raw-k-nvme Keep the complete raw-K authority in CPU memory.\n"
        "  --kvmem-prefix-cache  Serve plain (non-CB) route only: keep the shared\n"
        "                        executor warm across requests and prefill only the\n"
        "                        new suffix when a prompt strictly extends the prior\n"
        "                        request (prompt+response). Requires --kvmem.\n"
        "                        Default: off.\n"
        "  --kvmem-query-replay  Replay the final user query after query-conditioned\n"
        "                        block selection. Requires --kvmem and\n"
        "                        --kvmem-query-conditioned. Default: off.\n"
        "  --kvmem-guided-reselect M  Harness self-query trigger: off|boundary|\n"
        "                        middecode|both. Default: off.\n"
        "  --kvmem-guided-thinking-tokens N  Private planning cap. Default: 0\n"
        "                        (generate the contextual retrieval query directly).\n"
        "  --kvmem-guided-query-tokens N  Retrieval query cap, 1..512. Default: 256.\n"
        "  --kvmem-middecode-trigger-tokens N  Refresh threshold within each\n"
        "                        generation epoch. Default: 28672.\n"
        "  --kvmem-middecode-max-refreshes N  Per-request cap, 0..8. Default: 2.\n"
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
        "  --top-k N             Top-k. Default: 20 (Qwen3 recommended; 0 disables)\n"
        "  --min-p F             Min-p. Default: 0.0\n"
        "  --presence-penalty F  Presence penalty. Default: 0.0\n"
        "  --repetition-penalty F Repetition penalty. Default: 1.0\n"
        "  --seed N              Seed passed to llama.cpp\n"
        "\n"
        "kvmem-session (growth-profiling harness; one persistent process that\n"
        "  prefills a long context then keeps growing it across turns, measuring\n"
        "  the sequential wall-clock cost of every micro-step at each ladder point.\n"
        "  Forces --kvmem on, --kvmem-update-mode step, and MTP speculate on.\n"
        "  Reuses all --kvmem* and --kv-dtype flags; sizes --ctx automatically.):\n"
        "  --session-ladder L    Comma-separated cumulative context targets, e.g.\n"
        "                        256K,512K,1M,1.5M,2M. K/M/G suffixes + fractions\n"
        "                        allowed. Must be strictly increasing.\n"
        "                        Default: 256K,512K,1M,1.5M,2M.\n"
        "  --session-input FILE  Tokenize FILE once and use its exact prefixes as\n"
        "                        the ladder history. Default: synthetic corpus.\n"
        "  --session-decode-tokens N  MTP decode probe length per turn. Default: 256.\n"
        "  --session-query-tokens N  Tail tokens used as retrieval query at each\n"
        "                        ladder point. 0 disables query/replay. Default: 32.\n"
        "  --session-repeat-queries N  At each exact ladder checkpoint, issue N\n"
        "                        additional queries without re-prefilling history.\n"
        "                        0 keeps the legacy one-query path. Default: 0.\n"
        "  --session-repeat-mode M  Repeated-query state: frozen restores the same\n"
        "                        checkpoint before every query; sequential keeps\n"
        "                        prior probe turns. Default: frozen.\n"
        "  --session-prefill-probe-tokens N  Frozen-branch prefill chunk length at\n"
        "                        every milestone. 0 disables. Default: 0.\n"
        "  --session-prefill-probe-repeats N  Number of fixed prefill probes per\n"
        "                        milestone. 0 disables. Default: 0.\n"
        "  --temp F              Decode-probe temperature. Default 0 (greedy);\n"
        "                        --temp>0 uses the Qwen3 sampled recipe.\n"
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

uint64_t parse_gib_bytes(const std::string &s, const std::string &name) {
    size_t pos = 0;
    const double gib = std::stod(s, &pos);
    if (pos != s.size() || !std::isfinite(gib) || gib < 0.0) {
        throw std::runtime_error("invalid GiB value for " + name + ": " + s);
    }
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    const double bytes = gib * kGiB;
    if (bytes > static_cast<double>(std::numeric_limits<uint64_t>::max())) {
        throw std::runtime_error("GiB value too large for " + name + ": " + s);
    }
    return static_cast<uint64_t>(bytes);
}

// Parse a single token-count like "256K", "1M", "1.5M", "2097152". Suffix
// K/M/G are binary (1024-based); a bare number is taken as-is. Fractions are
// allowed with a suffix (e.g. "1.5M" = 1572864).
uint64_t parse_token_count(const std::string &s, const std::string &name) {
    if (s.empty()) throw std::runtime_error("empty token count for " + name);
    double mult = 1.0;
    std::string num = s;
    const char suf = static_cast<char>(std::toupper(s.back()));
    if (suf == 'K') { mult = 1024.0; num = s.substr(0, s.size() - 1); }
    else if (suf == 'M') { mult = 1024.0 * 1024.0; num = s.substr(0, s.size() - 1); }
    else if (suf == 'G') { mult = 1024.0 * 1024.0 * 1024.0; num = s.substr(0, s.size() - 1); }
    size_t pos = 0;
    const double v = std::stod(num, &pos);
    if (pos != num.size() || !std::isfinite(v) || v < 0.0) {
        throw std::runtime_error("invalid token count for " + name + ": " + s);
    }
    return static_cast<uint64_t>(v * mult);
}

// Parse a comma-separated cumulative ladder like "256K,512K,1M,1.5M,2M" into a
// strictly increasing list of token targets.
std::vector<uint64_t> parse_ladder(const std::string &s, const std::string &name) {
    std::vector<uint64_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        out.push_back(parse_token_count(item, name));
    }
    if (out.empty()) throw std::runtime_error("empty ladder for " + name);
    for (size_t i = 1; i < out.size(); ++i) {
        if (out[i] <= out[i - 1]) {
            throw std::runtime_error("ladder must be strictly increasing for " +
                                     name + ": " + s);
        }
    }
    return out;
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

    // `qw3 serve ...` runs the OpenAI/Anthropic-compatible HTTP server instead
    // of a one-shot generate. Detected as the first positional argument.
    bool serve = false;
    bool tokenize_only = false;
    qw3::ServerConfig serve_cfg;
    bool kv_dtype_cli_set = false;
    std::string kv_dtype_cli;

    // `qw3 kvmem-session ...` runs the kvmem context-growth profiling harness:
    // prefill a long context, then keep growing it across turns up to the
    // largest ladder target, printing the per-turn micro-step breakdown.
    bool kvmem_session = false;
    std::vector<uint64_t> session_ladder = {262144, 524288, 1048576, 1572864,
                                            2097152};
    int session_decode_tokens = 256;
    int session_query_tokens = 32;
    int session_repeat_queries = 0;
    std::string session_repeat_mode = "frozen";
    std::string session_input_path;
    int session_prefill_probe_tokens = 0;
    int session_prefill_probe_repeats = 0;

    // `qw3 archive build|query|info ...` drives the durable KVMem context
    // archive: ingest a corpus once, then attach it repeatedly at any prefix
    // length, under any retrieval policy, without recomputing the context.
    std::string archive_op;
    qw3::KvMemArchiveBuildConfig archive_build;
    qw3::KvMemArchiveRunConfig archive_run;
    std::string archive_questions_path;
    std::string archive_questions_json_path;
    std::string token_output_path;

    int arg_start = 1;
    if (argc > 1 && std::string(argv[1]) == "serve") {
        serve = true;
        arg_start = 2;
    } else if (argc > 1 && std::string(argv[1]) == "tokenize") {
        tokenize_only = true;
        arg_start = 2;
    } else if (argc > 1 && std::string(argv[1]) == "kvmem-session") {
        kvmem_session = true;
        arg_start = 2;
    } else if (argc > 1 && std::string(argv[1]) == "archive") {
        if (argc < 3) {
            std::cerr << "usage: qw3 archive build|query|info ...\n";
            return 2;
        }
        archive_op = argv[2];
        if (archive_op != "build" && archive_op != "query" &&
            archive_op != "info") {
            std::cerr << "unknown archive operation: " << archive_op
                      << " (expected build, query, or info)\n";
            return 2;
        }
        arg_start = 3;
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
            } else if (arg == "--native-kernels") {
                engine.native_kernels = need(arg);
            } else if (arg == "--native-linear-backend") {
                engine.native_linear_backend = need(arg);
            } else if (arg == "--cpu-embedding") {
                engine.cpu_embedding = true;
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
            } else if (arg == "--prefill-chunk") {
                engine.prefill_chunk = parse_int(need(arg), arg);
            } else if (arg == "--no-prefill-chunk") {
                engine.prefill_chunk = 0;
            } else if (arg == "--kvmem") {
                engine.kvmem_enabled = true;
            } else if (arg == "--kvmem-block-tokens") {
                engine.kvmem_block_tokens = parse_int(need(arg), arg);
            } else if (arg == "--kvmem-budget") {
                engine.kvmem_budget = parse_int(need(arg), arg);
            } else if (arg == "--kvmem-prefill-budget") {
                engine.kvmem_prefill_budget = parse_int(need(arg), arg);
            } else if (arg == "--kvmem-gen-budget") {
                engine.kvmem_gen_budget = parse_int(need(arg), arg);
            } else if (arg == "--kvmem-interval") {
                engine.kvmem_interval = parse_int(need(arg), arg);
            } else if (arg == "--kvmem-sink-tokens") {
                const int value = parse_int(need(arg), arg);
                if (value < 0) {
                    throw std::runtime_error(
                        "--kvmem-sink-tokens must be >= 0");
                }
                if (engine.kvmem_sink_blocks >= 0) {
                    throw std::runtime_error(
                        "--kvmem-sink-tokens cannot be combined with "
                        "--kvmem-sink-blocks");
                }
                engine.kvmem_sink_tokens = value;
            } else if (arg == "--kvmem-recent-tokens") {
                const int value = parse_int(need(arg), arg);
                if (value < 0) {
                    throw std::runtime_error(
                        "--kvmem-recent-tokens must be >= 0");
                }
                if (engine.kvmem_recent_blocks >= 0) {
                    throw std::runtime_error(
                        "--kvmem-recent-tokens cannot be combined with "
                        "--kvmem-recent-blocks");
                }
                engine.kvmem_recent_tokens = value;
            } else if (arg == "--kvmem-sink-blocks") {
                const int value = parse_int(need(arg), arg);
                if (value < 0) {
                    throw std::runtime_error(
                        "--kvmem-sink-blocks must be >= 0");
                }
                if (engine.kvmem_sink_tokens >= 0) {
                    throw std::runtime_error(
                        "--kvmem-sink-blocks cannot be combined with "
                        "--kvmem-sink-tokens");
                }
                engine.kvmem_sink_blocks = value;
            } else if (arg == "--kvmem-recent-blocks") {
                const int value = parse_int(need(arg), arg);
                if (value < 0) {
                    throw std::runtime_error(
                        "--kvmem-recent-blocks must be >= 0");
                }
                if (engine.kvmem_recent_tokens >= 0) {
                    throw std::runtime_error(
                        "--kvmem-recent-blocks cannot be combined with "
                        "--kvmem-recent-tokens");
                }
                engine.kvmem_recent_blocks = value;
            } else if (arg == "--kvmem-method") {
                engine.kvmem_method = need(arg);
                if (engine.kvmem_method != "retrieval" &&
                    engine.kvmem_method != "h2o" &&
                    engine.kvmem_method != "recency") {
                    throw std::runtime_error(
                        "--kvmem-method must be retrieval|h2o|recency");
                }
            } else if (arg == "--kvmem-select-policy") {
                engine.kvmem_select_policy = need(arg);
                if (engine.kvmem_select_policy != "topk" &&
                    engine.kvmem_select_policy != "quota") {
                    throw std::runtime_error(
                        "--kvmem-select-policy must be topk|quota");
                }
            } else if (arg == "--kvmem-retrieval-method") {
                engine.kvmem_retrieval_method = need(arg);
                if (engine.kvmem_retrieval_method != "mean-k" &&
                    engine.kvmem_retrieval_method != "per-token" &&
                    engine.kvmem_retrieval_method != "sub-block-mean-k" &&
                    engine.kvmem_retrieval_method !=
                        "key-direction-fixed4" &&
                    engine.kvmem_retrieval_method !=
                        "key-direction-adaptive") {
                    throw std::runtime_error(
                        "--kvmem-retrieval-method must be "
                        "mean-k|per-token|sub-block-mean-k|"
                        "key-direction-fixed4|key-direction-adaptive");
                }
            } else if (arg == "--kvmem-adaptive-gain-1to2") {
                engine.kvmem_adaptive_gain_1to2 =
                    parse_float(need(arg), arg);
                if (!std::isfinite(engine.kvmem_adaptive_gain_1to2) ||
                    engine.kvmem_adaptive_gain_1to2 < 0.0 ||
                    engine.kvmem_adaptive_gain_1to2 > 1.0) {
                    throw std::runtime_error(
                        "--kvmem-adaptive-gain-1to2 must be in [0,1]");
                }
            } else if (arg == "--kvmem-adaptive-gain-2to4") {
                engine.kvmem_adaptive_gain_2to4 =
                    parse_float(need(arg), arg);
                if (!std::isfinite(engine.kvmem_adaptive_gain_2to4) ||
                    engine.kvmem_adaptive_gain_2to4 < 0.0 ||
                    engine.kvmem_adaptive_gain_2to4 > 1.0) {
                    throw std::runtime_error(
                        "--kvmem-adaptive-gain-2to4 must be in [0,1]");
                }
            } else if (arg == "--kvmem-index-placement") {
                engine.kvmem_index_placement = need(arg);
                if (engine.kvmem_index_placement != "gpu" &&
                    engine.kvmem_index_placement != "cpu") {
                    throw std::runtime_error(
                        "--kvmem-index-placement must be gpu|cpu");
                }
            } else if (arg == "--kvmem-index-staging-mb") {
                engine.kvmem_index_staging_mb =
                    parse_int(need(arg), arg);
                if (engine.kvmem_index_staging_mb <= 0 ||
                    engine.kvmem_index_staging_mb > 4096) {
                    throw std::runtime_error(
                        "--kvmem-index-staging-mb must be in [1,4096]");
                }
            } else if (arg == "--kvmem-numa-policy") {
                engine.kvmem_numa_policy = need(arg);
                const std::string &policy = engine.kvmem_numa_policy;
                const bool node = policy.rfind("node:", 0) == 0 &&
                    policy.size() > 5 &&
                    std::all_of(policy.begin() + 5, policy.end(),
                                [](unsigned char ch) {
                                    return std::isdigit(ch) != 0;
                                });
                if (policy != "auto" && policy != "off" && !node) {
                    throw std::runtime_error(
                        "--kvmem-numa-policy must be auto|off|node:N");
                }
            } else if (arg == "--kvmem-adaptive-score-mode") {
                engine.kvmem_adaptive_score_mode = need(arg);
                if (engine.kvmem_adaptive_score_mode != "auto" &&
                    engine.kvmem_adaptive_score_mode !=
                        "layer-one-pass" &&
                    engine.kvmem_adaptive_score_mode !=
                        "tiled-one-pass" &&
                    engine.kvmem_adaptive_score_mode !=
                        "tiled-two-pass") {
                    throw std::runtime_error(
                        "--kvmem-adaptive-score-mode must be "
                        "auto|layer-one-pass|tiled-one-pass|tiled-two-pass");
                }
            } else if (arg == "--kvmem-subblocks") {
                engine.kvmem_subblocks = parse_int(need(arg), arg);
            } else if (arg == "--kvmem-subblock-reduce") {
                engine.kvmem_subblock_reduce = need(arg);
                if (engine.kvmem_subblock_reduce != "max" &&
                    engine.kvmem_subblock_reduce != "sum") {
                    throw std::runtime_error(
                        "--kvmem-subblock-reduce must be max|sum");
                }
            } else if (arg == "--kvmem-round-retrieval") {
                if (engine.kvmem_semantic_expansion != "none" &&
                    engine.kvmem_semantic_expansion != "round") {
                    throw std::runtime_error(
                        "--kvmem-round-retrieval conflicts with "
                        "--kvmem-semantic-expansion message");
                }
                engine.kvmem_semantic_expansion = "round";
            } else if (arg == "--kvmem-semantic-expansion") {
                engine.kvmem_semantic_expansion = need(arg);
                if (engine.kvmem_semantic_expansion != "none" &&
                    engine.kvmem_semantic_expansion != "round" &&
                    engine.kvmem_semantic_expansion != "message") {
                    throw std::runtime_error(
                        "--kvmem-semantic-expansion must be "
                        "none|round|message");
                }
            } else if (arg == "--kvmem-group-score-reduce") {
                engine.kvmem_group_score_reduce = need(arg);
                if (engine.kvmem_group_score_reduce != "max" &&
                    engine.kvmem_group_score_reduce !=
                        "length-normalized-mass") {
                    throw std::runtime_error(
                        "--kvmem-group-score-reduce must be "
                        "max|length-normalized-mass");
                }
            } else if (arg == "--kvmem-group-length-alpha") {
                engine.kvmem_group_length_alpha =
                    parse_float(need(arg), arg);
                if (engine.kvmem_group_length_alpha < 0.0 ||
                    engine.kvmem_group_length_alpha > 1.0) {
                    throw std::runtime_error(
                        "--kvmem-group-length-alpha must be in [0,1]");
                }
            } else if (arg == "--kvmem-update-mode") {
                engine.kvmem_update_mode = need(arg);
                if (engine.kvmem_update_mode != "interval" &&
                    engine.kvmem_update_mode != "step") {
                    throw std::runtime_error(
                        "--kvmem-update-mode must be interval|step");
                }
            } else if (arg == "--kvmem-opt-stage-out" ||
                       arg == "--kvmem-opt-stage-in" ||
                       arg == "--kvmem-opt-pack") {
                const std::string value = need(arg);
                if (value != "on" && value != "off") {
                    throw std::runtime_error(
                        arg + " must be on|off");
                }
                const bool enabled = value == "on";
                if (arg == "--kvmem-opt-stage-out") {
                    engine.kvmem_opt_stage_out = enabled;
                } else if (arg == "--kvmem-opt-stage-in") {
                    engine.kvmem_opt_stage_in = enabled;
                } else {
                    engine.kvmem_opt_pack = enabled;
                }
            } else if (arg == "--kvmem-query-conditioned") {
                engine.kvmem_query_conditioned = true;
            } else if (arg == "--kvmem-guided-reselect") {
                engine.kvmem_guided_reselect = need(arg);
                if (engine.kvmem_guided_reselect != "off" &&
                    engine.kvmem_guided_reselect != "boundary" &&
                    engine.kvmem_guided_reselect != "middecode" &&
                    engine.kvmem_guided_reselect != "both") {
                    throw std::runtime_error(
                        "--kvmem-guided-reselect must be "
                        "off|boundary|middecode|both");
                }
            } else if (arg == "--kvmem-guided-thinking-tokens") {
                engine.kvmem_guided_thinking_tokens =
                    parse_int(need(arg), arg);
                if (engine.kvmem_guided_thinking_tokens < 0 ||
                    engine.kvmem_guided_thinking_tokens > 4096) {
                    throw std::runtime_error(
                        "--kvmem-guided-thinking-tokens must be in [0,4096]");
                }
            } else if (arg == "--kvmem-guided-query-tokens") {
                engine.kvmem_guided_query_tokens = parse_int(need(arg), arg);
                if (engine.kvmem_guided_query_tokens < 1 ||
                    engine.kvmem_guided_query_tokens > 512) {
                    throw std::runtime_error(
                        "--kvmem-guided-query-tokens must be in [1,512]");
                }
            } else if (arg == "--kvmem-middecode-trigger-tokens") {
                engine.kvmem_middecode_trigger_tokens =
                    parse_int(need(arg), arg);
                if (engine.kvmem_middecode_trigger_tokens < 1) {
                    throw std::runtime_error(
                        "--kvmem-middecode-trigger-tokens must be positive");
                }
            } else if (arg == "--kvmem-middecode-max-refreshes") {
                engine.kvmem_middecode_max_refreshes =
                    parse_int(need(arg), arg);
                if (engine.kvmem_middecode_max_refreshes < 0 ||
                    engine.kvmem_middecode_max_refreshes > 8) {
                    throw std::runtime_error(
                        "--kvmem-middecode-max-refreshes must be in [0,8]");
                }
            } else if (arg == "--no-kvmem-recompute-query") {
                engine.kvmem_recompute_query = false;
            } else if (arg == "--kvmem-immutable-k") {
                engine.kvmem_immutable_source_k = true;
            } else if (arg == "--no-kvmem-immutable-k") {
                engine.kvmem_immutable_source_k = false;
            } else if (arg == "--kvmem-retrieval-blocks") {
                engine.kvmem_retrieval_blocks = parse_int(need(arg), arg);
            } else if (arg == "--kvmem-profile-blocks") {
                engine.kvmem_profile_blocks = parse_int(need(arg), arg);
            } else if (arg == "--kvmem-gpu-memory-ratio") {
                engine.kvmem_gpu_memory_ratio = parse_float(need(arg), arg);
            } else if (arg == "--kvmem-gpu-low-watermark") {
                engine.kvmem_gpu_low_watermark = parse_float(need(arg), arg);
            } else if (arg == "--kvmem-cpu-gb") {
                engine.kvmem_cpu_bytes = parse_gib_bytes(need(arg), arg);
            } else if (arg == "--kvmem-cpu-bytes") {
                engine.kvmem_cpu_bytes = parse_u64(need(arg), arg);
            } else if (arg == "--kvmem-nvme-dir") {
                engine.kvmem_nvme_dir = need(arg);
            } else if (arg == "--kvmem-nvme-gb") {
                engine.kvmem_nvme_bytes = parse_gib_bytes(need(arg), arg);
            } else if (arg == "--kvmem-nvme-bytes") {
                engine.kvmem_nvme_bytes = parse_u64(need(arg), arg);
            } else if (arg == "--kvmem-raw-k-nvme") {
                engine.kvmem_raw_k_nvme = true;
            } else if (arg == "--no-kvmem-raw-k-nvme") {
                engine.kvmem_raw_k_nvme = false;
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
                serve_cfg.temperature_set = true;
            } else if (arg == "--top-p") {
                gen.top_p = parse_float(need(arg), arg);
                serve_cfg.top_p_set = true;
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
            } else if (arg == "--prefix-cache") {
                serve_cfg.prefix_cache = true;
            } else if (arg == "--kvmem-prefix-cache") {
                serve_cfg.kvmem_prefix_cache = true;
            } else if (arg == "--kvmem-query-replay") {
                serve_cfg.kvmem_query_replay = true;
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
            } else if (arg == "--kvmem-archive") {
                engine.kvmem_archive_dir = need(arg);
            } else if (arg == "--archive-input") {
                archive_build.input_path = need(arg);
            } else if (arg == "--archive-token-input") {
                archive_build.token_input_path = need(arg);
            } else if (arg == "--archive-tokens") {
                // Build: how much of the corpus to ingest.
                // Query: which prefix of the archive to attach.
                const uint64_t v = std::stoull(need(arg));
                archive_build.tokens = v;
                archive_run.tokens = v;
                engine.kvmem_archive_tokens = v;
            } else if (arg == "--archive-ladder-tokens") {
                archive_build.ladder_tokens = std::stoull(need(arg));
            } else if (arg == "--archive-pad-final-chunk") {
                archive_build.pad_final_chunk = true;
            } else if (arg == "--archive-prefill-window") {
                const std::string mode = need(arg);
                if (mode == "pressure") {
                    archive_build.prefill_window =
                        qw3::KvMemArchiveBuildConfig::PrefillWindow::Pressure;
                } else if (mode == "semantic_chunk") {
                    archive_build.prefill_window = qw3::KvMemArchiveBuildConfig::
                        PrefillWindow::SemanticChunk;
                } else {
                    throw std::runtime_error(
                        "--archive-prefill-window must be "
                        "pressure|semantic_chunk");
                }
            } else if (arg ==
                       "--archive-prefill-semantic-start-tokens") {
                const int value = parse_int(need(arg), arg);
                if (value < 0) {
                    throw std::runtime_error(
                        "--archive-prefill-semantic-start-tokens must be >= 0");
                }
                archive_build.semantic_start_tokens =
                    static_cast<uint32_t>(value);
            } else if (arg ==
                       "--archive-prefill-semantic-query-tokens") {
                const int value = parse_int(need(arg), arg);
                if (value < 0) {
                    throw std::runtime_error(
                        "--archive-prefill-semantic-query-tokens must be >= 0");
                }
                archive_build.semantic_query_tokens =
                    static_cast<uint32_t>(value);
            } else if (arg == "--archive-question") {
                archive_run.questions.push_back({need(arg)});
            } else if (arg == "--archive-questions-file") {
                archive_questions_path = need(arg);
            } else if (arg == "--archive-questions-json") {
                archive_questions_json_path = need(arg);
            } else if (arg == "--archive-question-format") {
                const std::string format = need(arg);
                if (format == "raw") {
                    archive_run.question_format =
                        qw3::KvMemArchiveRunConfig::QuestionFormat::Raw;
                } else if (format == "qwen-chat") {
                    archive_run.question_format = qw3::KvMemArchiveRunConfig::
                        QuestionFormat::QwenChatThinking;
                } else if (format == "qwen-chat-no-thinking") {
                    archive_run.question_format = qw3::KvMemArchiveRunConfig::
                        QuestionFormat::QwenChatNoThinking;
                } else if (format == "qwen-user-continuation") {
                    archive_run.question_format = qw3::KvMemArchiveRunConfig::
                        QuestionFormat::QwenUserContinuationNoThinking;
                } else {
                    throw std::runtime_error(
                        "--archive-question-format must be "
                        "raw|qwen-chat|qwen-chat-no-thinking|"
                        "qwen-user-continuation");
                }
            } else if (arg == "--archive-query-token-selector") {
                const std::string selector = need(arg);
                if (selector == "none") {
                    archive_run.query_attention_probe = false;
                    archive_run.query_guided_query = false;
                } else if (selector == "attention-probe") {
                    archive_run.query_attention_probe = true;
                    archive_run.query_guided_query = false;
                } else if (selector == "guided-query") {
                    archive_run.query_attention_probe = false;
                    archive_run.query_guided_query = true;
                } else {
                    throw std::runtime_error(
                        "--archive-query-token-selector must be "
                        "none|attention-probe|guided-query");
                }
            } else if (arg == "--archive-query-probe-tokens") {
                const int value = parse_int(need(arg), arg);
                if (value <= 0) {
                    throw std::runtime_error(
                        "--archive-query-probe-tokens must be > 0");
                }
                archive_run.query_probe_tokens =
                    static_cast<uint32_t>(value);
            } else if (arg == "--archive-query-score-tokens") {
                const int value = parse_int(need(arg), arg);
                if (value <= 0) {
                    throw std::runtime_error(
                        "--archive-query-score-tokens must be > 0");
                }
                archive_run.query_score_tokens =
                    static_cast<uint32_t>(value);
            } else if (arg == "--archive-results-file") {
                archive_run.results_path = need(arg);
            } else if (arg == "--token-output") {
                token_output_path = need(arg);
            } else if (arg == "--session-ladder") {
                session_ladder = parse_ladder(need(arg), arg);
            } else if (arg == "--session-input") {
                session_input_path = need(arg);
            } else if (arg == "--session-decode-tokens") {
                session_decode_tokens = parse_int(need(arg), arg);
                if (session_decode_tokens <= 0) {
                    throw std::runtime_error("--session-decode-tokens must be > 0");
                }
            } else if (arg == "--session-query-tokens") {
                session_query_tokens = parse_int(need(arg), arg);
                if (session_query_tokens < 0) {
                    throw std::runtime_error(
                        "--session-query-tokens must be >= 0");
                }
            } else if (arg == "--session-repeat-queries") {
                session_repeat_queries = parse_int(need(arg), arg);
                if (session_repeat_queries < 0) {
                    throw std::runtime_error(
                        "--session-repeat-queries must be >= 0");
                }
            } else if (arg == "--session-repeat-mode") {
                session_repeat_mode = need(arg);
                if (session_repeat_mode != "frozen" &&
                    session_repeat_mode != "sequential") {
                    throw std::runtime_error(
                        "--session-repeat-mode must be frozen|sequential");
                }
            } else if (arg == "--session-prefill-probe-tokens") {
                session_prefill_probe_tokens = parse_int(need(arg), arg);
                if (session_prefill_probe_tokens < 0) {
                    throw std::runtime_error(
                        "--session-prefill-probe-tokens must be >= 0");
                }
            } else if (arg == "--session-prefill-probe-repeats") {
                session_prefill_probe_repeats = parse_int(need(arg), arg);
                if (session_prefill_probe_repeats < 0) {
                    throw std::runtime_error(
                        "--session-prefill-probe-repeats must be >= 0");
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

        if (tokenize_only) {
            if (engine.model_path.empty()) {
                throw std::runtime_error("qw3 tokenize requires --model");
            }
            if (prompt.empty()) {
                throw std::runtime_error(
                    "qw3 tokenize requires --prompt or --prompt-file");
            }
            const qw3::GgufFile gguf(engine.model_path);
            const qw3::QwenTokenizer tokenizer(gguf);
            const std::vector<int32_t> tokens = tokenizer.encode(prompt, false);
            if (!token_output_path.empty()) {
                std::ofstream out(token_output_path,
                                  std::ios::out | std::ios::binary |
                                      std::ios::trunc);
                if (!out) {
                    throw std::runtime_error("cannot create token output: " +
                                             token_output_path);
                }
                static constexpr char magic[8] = {
                    'Q', 'W', '3', 'T', 'O', 'K', '2', '\0'};
                const uint64_t count = tokens.size();
                const std::string model_sha256 =
                    qw3::kvmem_archive_model_sha256(engine.model_path);
                if (model_sha256.size() != 64) {
                    throw std::runtime_error(
                        "failed to identify tokenizer model for token output");
                }
                out.write(magic, sizeof(magic));
                out.write(reinterpret_cast<const char *>(&count),
                          sizeof(count));
                out.write(model_sha256.data(),
                          static_cast<std::streamsize>(model_sha256.size()));
                if (!tokens.empty()) {
                    out.write(reinterpret_cast<const char *>(tokens.data()),
                              static_cast<std::streamsize>(
                                  tokens.size() * sizeof(tokens[0])));
                }
                if (!out) {
                    throw std::runtime_error("failed writing token output: " +
                                             token_output_path);
                }
            }
            std::cout << "tokens=" << tokens.size()
                      << " bytes=" << prompt.size() << "\n";
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
            if (!engine.kvmem_archive_dir.empty()) {
                if (kv_dtype_cli_set && kv_dtype_cli != "fp8") {
                    throw std::runtime_error(
                        "archive-backed serving requires --kv-dtype fp8");
                }
                if (serve_cfg.continuous_batching) {
                    throw std::runtime_error(
                        "archive-backed serving currently uses the serialized "
                        "frozen-branch path and cannot use --continuous-batching");
                }
                engine.kvmem_enabled = true;
                engine.kvmem_immutable_source_k = true;
                engine.kvmem_raw_k_nvme = true;
                engine.kvmem_query_conditioned = true;
                engine.kvmem_update_mode = "step";
                engine.kvmem_archive_mode = "attach";
                if (!engine.native_mtp_chain_set ||
                    engine.native_mtp_chain <= 0) {
                    engine.native_mtp_chain = 4;
                    engine.native_mtp_chain_set = true;
                }
                engine.native_mtp_speculate = true;
                serve_cfg.kv_dtype = "fp8";
            }
            serve_cfg.default_generation = gen;
            return qw3::run_server(engine, serve_cfg);
        }

        if (!archive_op.empty()) {
            if (archive_op == "info") {
                if (engine.kvmem_archive_dir.empty()) {
                    throw std::runtime_error(
                        "qw3 archive info requires --kvmem-archive");
                }
                return qw3::run_kvmem_archive_info(engine.kvmem_archive_dir);
            }
            if (kv_dtype_cli_set && kv_dtype_cli != "fp8") {
                throw std::runtime_error(
                    "the KVMem context archive format is fp8-only; "
                    "--kv-dtype " + kv_dtype_cli + " is unsupported");
            }
            if (archive_op == "build") {
                if (engine.ctx_size <= 0) {
                    throw std::runtime_error(
                        "qw3 archive build requires --ctx large enough for the "
                        "full context");
                }
                if (archive_build.prefill_window !=
                        qw3::KvMemArchiveBuildConfig::PrefillWindow::
                            SemanticChunk &&
                    (archive_build.semantic_start_tokens != 0 ||
                     archive_build.semantic_query_tokens != 0)) {
                    throw std::runtime_error(
                        "--archive-prefill-semantic-* requires "
                        "--archive-prefill-window semantic_chunk");
                }
                return qw3::run_kvmem_archive_build(engine, archive_build);
            }
            if (!archive_questions_path.empty()) {
                std::ifstream qin(archive_questions_path);
                if (!qin) {
                    throw std::runtime_error("cannot read " +
                                             archive_questions_path);
                }
                std::string line;
                while (std::getline(qin, line)) {
                    if (!line.empty()) archive_run.questions.push_back({line});
                }
            }
            if (!archive_questions_json_path.empty()) {
                std::ifstream qin(archive_questions_json_path);
                if (!qin) {
                    throw std::runtime_error("cannot read " +
                                             archive_questions_json_path);
                }
                nlohmann::json questions;
                qin >> questions;
                if (!questions.is_array()) {
                    throw std::runtime_error(
                        "--archive-questions-json must contain a JSON array");
                }
                for (const auto &question : questions) {
                    if (question.is_string()) {
                        archive_run.questions.push_back(
                            {question.get<std::string>()});
                        continue;
                    }
                    if (!question.is_object() ||
                        !question.contains("content") ||
                        !question["content"].is_string() ||
                        !question.contains("query_content_start") ||
                        !question["query_content_start"].is_number_integer() ||
                        !question.contains("query_content_end") ||
                        !question["query_content_end"].is_number_integer()) {
                        throw std::runtime_error(
                            "--archive-questions-json entries must be strings "
                            "or objects with string content and integer "
                            "query_content_start/query_content_end");
                    }
                    const std::string content =
                        question["content"].get<std::string>();
                    const int64_t begin =
                        question["query_content_start"].get<int64_t>();
                    const int64_t end =
                        question["query_content_end"].get<int64_t>();
                    if (begin < 0 || end <= begin ||
                        static_cast<uint64_t>(end) > content.size()) {
                        throw std::runtime_error(
                            "--archive-questions-json query content byte "
                            "offsets are empty or outside content");
                    }
                    const auto is_utf8_boundary = [&](size_t offset) {
                        return offset == content.size() ||
                            (static_cast<unsigned char>(content[offset]) &
                             0xc0u) != 0x80u;
                    };
                    if (!is_utf8_boundary(static_cast<size_t>(begin)) ||
                        !is_utf8_boundary(static_cast<size_t>(end))) {
                        throw std::runtime_error(
                            "--archive-questions-json query content offsets "
                            "must be UTF-8 byte boundaries");
                    }
                    qw3::KvMemArchiveRunConfig::Question parsed;
                    parsed.content = content;
                    parsed.query_content_start = static_cast<uint64_t>(begin);
                    parsed.query_content_end = static_cast<uint64_t>(end);
                    parsed.has_query_content_span = true;
                    archive_run.questions.push_back(std::move(parsed));
                }
            }
            if (archive_run.questions.empty()) {
                throw std::runtime_error(
                    "qw3 archive query requires --archive-question or "
                    "--archive-questions-file/--archive-questions-json");
            }
            archive_run.max_tokens = gen.max_tokens;
            archive_run.thinking_budget =
                serve_cfg.thinking_budget_default;
            if (serve_cfg.temperature_set) {
                archive_run.temperature = gen.temperature;
                archive_run.top_p = gen.top_p;
                archive_run.top_k = gen.top_k;
            }
            if (engine.ctx_size <= 0) {
                const qw3::KvMemArchiveManifest m =
                    qw3::KvMemArchive::read_manifest(engine.kvmem_archive_dir);
                engine.ctx_size = static_cast<int>(
                    m.total_tokens + 4096 +
                    static_cast<uint64_t>(std::max(1, gen.max_tokens)));
            }
            return qw3::run_kvmem_archive_query(engine, archive_run);
        }

        if (kvmem_session) {
            engine.backend = qw3::BackendKind::QwenNative;
            engine.native_heavy = true;
            if (engine.native_kernels.empty()) engine.native_kernels = "cuda";
            if (kv_dtype_cli_set) {
                setenv("QW3_KV_DTYPE", kv_dtype_cli.c_str(), 1);
            }
            // ctx_size must cover the largest ladder target plus the decode
            // probes so the executor's KV cache can hold the grown context.
            const uint64_t max_target = session_ladder.back();
            const uint64_t need_ctx =
                max_target +
                static_cast<uint64_t>(session_decode_tokens) *
                    session_ladder.size() +
                static_cast<uint64_t>(session_repeat_queries) *
                    static_cast<uint64_t>(session_query_tokens +
                                          session_decode_tokens) +
                static_cast<uint64_t>(session_prefill_probe_tokens) +
                4096;
            if (engine.ctx_size <= 0 ||
                static_cast<uint64_t>(engine.ctx_size) < need_ctx) {
                engine.ctx_size = static_cast<int>(need_ctx);
            }
            qw3::KvMemSessionConfig sess;
            sess.ladder_tokens = session_ladder;
            sess.input_path = session_input_path;
            sess.decode_tokens = session_decode_tokens;
            sess.query_tokens = session_query_tokens;
            sess.repeat_queries = session_repeat_queries;
            sess.repeat_mode = session_repeat_mode;
            sess.prefill_probe_tokens = session_prefill_probe_tokens;
            sess.prefill_probe_repeats = session_prefill_probe_repeats;
            if (session_repeat_queries > 0 && session_query_tokens <= 0) {
                throw std::runtime_error(
                    "--session-repeat-queries requires "
                    "--session-query-tokens > 0");
            }
            if (session_query_tokens > 0) {
                engine.kvmem_query_conditioned = true;
                setenv("QW3_KVMEM_QUERY_REPLAY", "1", 1);
            }
            // Default greedy; --temp opts the decode probe into the Qwen3
            // sampled recipe (top_p/top_k carry their gen defaults 0.95/20).
            if (serve_cfg.temperature_set) {
                sess.temperature = gen.temperature;
                sess.top_p = gen.top_p;
                sess.top_k = gen.top_k;
            }
            return qw3::run_kvmem_session(engine, sess);
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
