#pragma once
#include "qw3/kvmem_mismatch.hpp"
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>

namespace qw3::detail {
// Immutable, position-free Mean-K: [sampled layer, KV head, head dimension].
// Producer replaces changed blocks; a worker never reads the live executor.
struct MismatchKeyBlock {
    std::vector<uint16_t> keys;
    // Optional immutable raw rows supplement the live index's decode lag.
    // [layer, token, KV head, dimension], native unscaled FP8/F16/F32.
    std::vector<uint8_t> raw;
    uint32_t raw_tokens = 0, raw_elem_bytes = 0;
};
struct MismatchMeanQuery {
    uint64_t position = 0;
    uint32_t tokens = 0;
    std::vector<float> values;
};
// Carries a partial block across accepted decode batches. No rejected rows
// enter the sum; discontinuities discard the old partial, never blend contexts.
struct MismatchQueryMean {
    uint64_t next = 0, discarded_tokens = 0;
    uint32_t count = 0;
    std::vector<float> sum;
    void reset();
    std::vector<MismatchMeanQuery> append(const uint16_t *queries, uint32_t layers,
        uint32_t captured_rows, uint32_t accepted_rows, uint32_t width,
        uint64_t base, uint32_t block_tokens, uint32_t first_row = 0);
};
struct MismatchJob {
    uint64_t epoch = 0, selection = 0, position = 0, progress = 0;
    uint64_t indexed_tokens = 0, tail_begin = 0, skipped_busy = 0;
    uint32_t block_tokens = 0, heads = 0, kv_heads = 0, dim = 0, sink_blocks = 0;
    bool prefill = false;
    double capture_ms = 0, snapshot_ms = 0, index_update_ms_total = 0;
    uint64_t index_copy_bytes_total = 0;
    std::vector<int32_t> layers;
    std::vector<uint64_t> query_positions;
    // [layer, query row, Query head, dimension], FP16 content frame.
    std::vector<uint16_t> queries;
    std::vector<float> mean_queries;
    std::vector<uint32_t> query_token_counts;
    uint64_t official_indexed_tokens = 0, skipped_query_blocks = 0;
    uint64_t discarded_query_tokens = 0;
    std::vector<std::shared_ptr<const MismatchKeyBlock>> blocks;
    // Number of tokens retained from each block's beginning in the selected
    // prefix. Dense appended tail is represented separately, never double counted.
    std::vector<uint32_t> selected_tokens;
    // Policy-forced prefix tokens in each historical block. Empty means no
    // explicit mandatory prefix; otherwise each entry must be a selected subset.
    // The observer unions this prefix with the accepted dense tail for its
    // token-relaxed random reference. This does not change the selected window.
    std::vector<uint32_t> mandatory_tokens;
    std::vector<uint8_t> live;
};
struct MismatchScore {
    double missing = 0, min_missing = 1, max_missing = 0;
    std::vector<double> query_coverage;
    // All query arrays align with MismatchJob::query_positions. Candidate tokens
    // count only live, causally complete blocks, including any raw supplement.
    std::vector<uint64_t> query_candidate_tokens, query_visible_tokens, query_mandatory_tokens;
    std::vector<double> query_mandatory_coverage, query_random_coverage, query_lift;
    std::vector<uint8_t> query_baseline_valid;
    std::vector<std::string> query_baseline_status;
    // Same candidate-valid row average as missing; lift is the ratio of these
    // job averages, never the average of per-query lift. Undefined values are NaN.
    double random_coverage = std::numeric_limits<double>::quiet_NaN();
    double lift = std::numeric_limits<double>::quiet_NaN();
    uint64_t distributions = 0, candidate_blocks = 0;
};
float mismatch_half_to_float(uint16_t value);
double mismatch_decay(double previous, double missing, double epsilon,
                      uint64_t delta_tokens, double half_life);
MismatchScore score_mismatch(const MismatchJob &job,
    std::vector<std::shared_ptr<const MismatchKeyBlock>> &cache_ids,
    std::vector<std::vector<float>> &cache_values);

// At most one job (including in-flight work). A busy observer drops a probe,
// never queues an unbounded backlog or blocks inference waiting for scoring.
class MismatchObserver {
public:
    explicit MismatchObserver(KvMemMismatchConfig config);
    ~MismatchObserver();
    MismatchObserver(const MismatchObserver &) = delete;
    MismatchObserver &operator=(const MismatchObserver &) = delete;
    bool busy() const { return busy_.load(std::memory_order_acquire); }
    bool submit(MismatchJob job);
    void finish(); // Only at teardown / tests, not on the token path.
private:
    void run();
    KvMemMismatchConfig config_;
    uint64_t id_ = 0;
    std::ofstream log_;
    std::atomic<bool> busy_{false};
    std::mutex mutex_;
    std::condition_variable wake_;
    std::unique_ptr<MismatchJob> pending_;
    bool stopping_ = false;
    std::thread worker_;
};
}
