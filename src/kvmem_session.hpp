#pragma once

#include "qw3/qw3.hpp"

#include <cstdint>
#include <vector>

namespace qw3 {

// Configuration for the kvmem growth-profiling harness. Drives one persistent
// process that prefills an initial long context and then keeps growing it
// across "turns" (each turn appends a fresh document chunk and decodes a short
// probe), measuring the sequential wall-clock cost of every micro-step
// (selection -> stage-in -> stage-out -> assemble -> prefill -> decode) at each
// ladder point. kvmem + step update-mode + MTP are forced on by run_kvmem_session.
struct KvMemSessionConfig {
    // Cumulative context targets in tokens, e.g. {262144, 524288, 1048576,
    // 1572864, 2097152}. Each turn prefills the delta needed to bring the
    // running position up to the next target before the decode probe.
    std::vector<uint64_t> ladder_tokens;
    // Tokens to decode (MTP) at each ladder point to sample steady-state TBT.
    int decode_tokens = 256;
};

// Loads the model once (kvmem on, update-mode step, MTP on) and runs the growth
// ladder, printing the per-turn micro-step breakdown and a final summary table.
// Returns a process exit code (0 on success). Blocking.
int run_kvmem_session(EngineOptions engine, const KvMemSessionConfig &cfg);

} // namespace qw3
