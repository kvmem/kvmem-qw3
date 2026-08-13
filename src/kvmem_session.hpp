#pragma once

#include "qw3/qw3.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace qw3 {

// Configuration for the kvmem growth-profiling harness. Drives one persistent
// process that prefills an initial long context and then keeps growing it
// across "turns" (each turn appends a fresh document chunk and decodes a short
// probe), measuring mutually exclusive top-level phases at each ladder point.
// Nested selection/I/O/assembly counters are also reported, but may overlap.
// kvmem + step update-mode + MTP are forced on by run_kvmem_session.
struct KvMemSessionConfig {
    // Cumulative context targets in tokens, e.g. {262144, 524288, 1048576,
    // 1572864, 2097152}. Each turn prefills the delta needed to bring the
    // running position up to the next target before the decode probe.
    std::vector<uint64_t> ladder_tokens;
    // Optional real history corpus. When set, the file is tokenized once and
    // exact prefix slices drive the ladder instead of the synthetic corpus.
    std::string input_path;
    // Tokens to decode (MTP) at each ladder point to sample steady-state TBT.
    int decode_tokens = 256;
    // Final tokens of each synthetic append that act as an absolute-position
    // retrieval query. Positive values enable query-conditioned selection and
    // block-aligned query replay at every above-budget ladder point. Zero keeps
    // the legacy fallback-score profiler.
    int query_tokens = 32;
    // Optional milestone fan-out. When positive, each ladder target is first
    // ingested with max_tokens=0 and captured at the exact target position.
    // The harness then issues this many synthetic queries from that checkpoint.
    // "frozen" restores the same checkpoint before every query; "sequential"
    // lets queries observe earlier probe turns. In both modes the checkpoint is
    // restored after the probes so the next ladder delta remains exact.
    int repeat_queries = 0;
    std::string repeat_mode = "frozen"; // frozen|sequential
    // Fixed frozen-branch prefill probes at every milestone. Each probe
    // restores the milestone, appends the same token chunk with final semantic
    // re-selection disabled, records prefill_s, then restores again.
    int prefill_probe_tokens = 0;
    int prefill_probe_repeats = 0;
    // Decode-probe sampling. Default is greedy (temp=0) for a stable
    // steady-state throughput probe; --temp/--top-p/--top-k route the Qwen3
    // sampled path (MTP is distribution-lossless under temp>0).
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 0;
};

// Loads the model once (kvmem on, update-mode step, MTP on) and runs the growth
// ladder, printing the per-turn micro-step breakdown and a final summary table.
// Returns a process exit code (0 on success). Blocking.
int run_kvmem_session(EngineOptions engine, const KvMemSessionConfig &cfg);

} // namespace qw3
