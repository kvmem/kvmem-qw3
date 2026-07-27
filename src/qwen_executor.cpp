#include "qwen_executor.hpp"
#include "env_flags.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace qw3 {

class KvmemCpuWorkerPool {
public:
    explicit KvmemCpuWorkerPool(size_t helper_count) {
        threads_.reserve(helper_count);
        for (size_t index = 0; index < helper_count; ++index) {
            threads_.emplace_back([this, index]() { worker_loop(index); });
        }
    }

    ~KvmemCpuWorkerPool() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
            ++generation_;
        }
        work_cv_.notify_all();
        for (std::thread &thread : threads_) {
            if (thread.joinable()) thread.join();
        }
    }

    KvmemCpuWorkerPool(const KvmemCpuWorkerPool &) = delete;
    KvmemCpuWorkerPool &operator=(const KvmemCpuWorkerPool &) = delete;

    size_t max_workers() const { return threads_.size() + 1; }

    void run(size_t task_count, size_t requested_workers,
             const std::function<void(size_t)> &task) {
        if (task_count == 0) return;
        const size_t workers = std::min(
            {requested_workers, task_count, max_workers()});
        if (workers <= 1) {
            for (size_t i = 0; i < task_count; ++i) task(i);
            return;
        }

        // Executor calls are presently serialized, but retain an explicit run
        // mutex so future prefetch/assembly overlap cannot publish two jobs
        // into the same pool concurrently.
        std::lock_guard<std::mutex> run_lock(run_mu_);
        {
            std::lock_guard<std::mutex> lock(mu_);
            task_ = task;
            task_count_ = task_count;
            next_task_.store(0, std::memory_order_relaxed);
            active_helpers_ = workers - 1;
            completed_helpers_ = 0;
            error_ = nullptr;
            ++generation_;
        }
        work_cv_.notify_all();
        consume_tasks();

        std::exception_ptr error;
        {
            std::unique_lock<std::mutex> lock(mu_);
            done_cv_.wait(lock, [&]() {
                return completed_helpers_ == active_helpers_;
            });
            error = error_;
            task_ = {};
        }
        if (error) std::rethrow_exception(error);
    }

private:
    void record_error(std::exception_ptr error) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!error_) error_ = std::move(error);
        next_task_.store(task_count_, std::memory_order_relaxed);
    }

    void consume_tasks() {
        try {
            for (;;) {
                const size_t index =
                    next_task_.fetch_add(1, std::memory_order_relaxed);
                if (index >= task_count_) break;
                task_(index);
            }
        } catch (...) {
            record_error(std::current_exception());
        }
    }

    void worker_loop(size_t helper_index) {
        uint64_t observed_generation = 0;
        std::unique_lock<std::mutex> lock(mu_);
        for (;;) {
            work_cv_.wait(lock, [&]() {
                return stop_ || generation_ != observed_generation;
            });
            if (stop_) return;
            observed_generation = generation_;
            const bool active = helper_index < active_helpers_;
            lock.unlock();
            if (active) consume_tasks();
            lock.lock();
            if (active) {
                ++completed_helpers_;
                if (completed_helpers_ == active_helpers_) {
                    done_cv_.notify_one();
                }
            }
        }
    }

    std::mutex run_mu_;
    std::mutex mu_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    std::vector<std::thread> threads_;
    std::function<void(size_t)> task_;
    std::atomic<size_t> next_task_{0};
    size_t task_count_ = 0;
    size_t active_helpers_ = 0;
    size_t completed_helpers_ = 0;
    uint64_t generation_ = 0;
    bool stop_ = false;
    std::exception_ptr error_;
};

namespace {

void require_status(const DeviceStatus &st) {
    if (!st.ok) throw std::runtime_error(st.message);
}

bool scorer_unavailable(std::string *failure_reason, const char *reason) {
    if (failure_reason) *failure_reason = reason;
    return false;
}

bool scorer_backend_unavailable(std::string *failure_reason,
                                const char *reason,
                                const DeviceStatus &status) {
    if (!failure_reason) return false;
    *failure_reason = reason;
    if (!status.message || !*status.message) return false;
    failure_reason->push_back('(');
    constexpr size_t kMaxDetail = 160;
    size_t emitted = 0;
    for (const char *p = status.message; *p && emitted < kMaxDetail; ++p) {
        const char ch = *p;
        const unsigned char u = static_cast<unsigned char>(ch);
        const bool keep = (u >= 'a' && u <= 'z') ||
                          (u >= 'A' && u <= 'Z') ||
                          (u >= '0' && u <= '9') ||
                          ch == '.' || ch == '_' || ch == '-' ||
                          ch == '/' || ch == ':';
        failure_reason->push_back(keep ? ch : '_');
        ++emitted;
    }
    failure_reason->push_back(')');
    return false;
}

uint64_t tensor_rows(const GgufTensorInfo &tensor) {
    if (tensor.dims.size() < 2) return 1;
    uint64_t rows = 1;
    for (size_t i = 1; i < tensor.dims.size(); ++i) rows *= tensor.dims[i];
    return rows;
}

double steady_seconds() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

bool executor_trace_timing_enabled() {
    return env_flag_enabled("QW3_DECODE_TRACE") ||
           env_flag_enabled("QW3_EXECUTOR_TRACE") ||
           env_flag_enabled("QW3_MTP_VERIFY_TRACE");
}

bool full_executor_trace_enabled() {
    return env_flag_enabled("QW3_EXECUTOR_TRACE") ||
           env_flag_enabled("QW3_MTP_VERIFY_TRACE");
}

bool paged_kv_prefill_for_local_cache_enabled() {
    return env_flag_enabled("QW3_PAGED_KV_PREFILL", true);
}

const char *kvmem_optimization_level_name(KvMemOptimizationLevel level) {
    switch (level) {
        case KvMemOptimizationLevel::KvmemInit: return "kvmem_init";
        case KvMemOptimizationLevel::Opt1: return "opt_1";
        case KvMemOptimizationLevel::Opt2: return "opt_2";
        case KvMemOptimizationLevel::Opt3: return "opt_3";
    }
    return "unknown";
}

size_t kvmem_env_size(const char *name, size_t fallback,
                      size_t minimum, size_t maximum) {
    const char *value = std::getenv(name);
    if (!value || !*value) return fallback;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') return fallback;
    return std::max(minimum, std::min(
        maximum, static_cast<size_t>(parsed)));
}

size_t kvmem_io_slab_bytes() {
    static const size_t value =
        kvmem_env_size("QW3_KVMEM_IO_SLAB_MIB", 128, 4, 256) *
        1024ull * 1024ull;
    return value;
}

size_t kvmem_write_queue_depth() {
    static const size_t value =
        kvmem_env_size("QW3_KVMEM_WRITE_QUEUE_DEPTH", 8, 1, 32);
    return value;
}

size_t kvmem_writeback_slab_count() {
    // Four bounded slabs provide enough queue depth for alternating writeback
    // buffer sets. At the 128 MiB default, immutable MAIN V plus local-position
    // MTP V for a 2048-token chunk (~68 MiB) fits in one slab; smaller
    // QW3_KVMEM_IO_SLAB_MIB settings retain the multi-slab compatibility path.
    static const size_t value =
        kvmem_env_size("QW3_KVMEM_WRITEBACK_SLABS", 4, 2, 16);
    return value;
}

size_t kvmem_cpu_gather_threads() {
    static const size_t value =
        kvmem_env_size("QW3_KVMEM_CPU_GATHER_THREADS", 4, 1, 16);
    return value;
}

size_t kvmem_overlap_stagein_threads() {
    // V comes from sparse CPU spill slabs and is measurably less bandwidth
    // efficient than block-major raw K. During overlap it is normally the
    // critical branch, so give it more workers than the raw-K gather.
    static const size_t value =
        kvmem_env_size(
            "QW3_KVMEM_OVERLAP_STAGEIN_THREADS", 8, 1, 16);
    return value;
}

size_t kvmem_cpu_stageout_threads() {
    static const size_t value =
        kvmem_env_size("QW3_KVMEM_CPU_STAGEOUT_THREADS", 4, 1, 16);
    return value;
}

// Diagnostic-only guardrail for finding every execution path that feeds a
// position outside the model's trained RoPE range into a RoPE/de-RoPE kernel.
// It deliberately does not fail or change the position: the point of
// QW3_ROPE_POSITION_TRACE=1 is to let a long request finish and produce a
// complete source-labelled inventory before changing any position policy.
void trace_rope_position_if_out_of_range(const char *source,
                                         int64_t base_pos,
                                         uint32_t n_tokens,
                                         uint32_t limit,
                                         int32_t layer = -1,
                                         uint32_t kernel_uses = 1) {
    static const bool enabled = env_flag_enabled("QW3_ROPE_POSITION_TRACE");
    if (!enabled || n_tokens == 0 || limit == 0) return;
    const int64_t end_pos = base_pos + static_cast<int64_t>(n_tokens);
    if (base_pos >= 0 && end_pos <= static_cast<int64_t>(limit)) return;
    static std::atomic<uint64_t> sequence{0};
    const uint64_t seq = sequence.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr,
                 "[rope-position-oob] seq=%llu source=%s base=%lld "
                 "end_exclusive=%lld tokens=%u limit=%u layer=%d uses=%u\n",
                 static_cast<unsigned long long>(seq), source,
                 static_cast<long long>(base_pos),
                 static_cast<long long>(end_pos), n_tokens, limit, layer,
                 kernel_uses);
}

struct QuerySpanOverlap {
    uint32_t row_offset = 0;
    uint32_t count = 0;
};

// All query spans supplied by the serve layer are absolute token positions in
// the complete rendered prompt. `chunk_begin` is also absolute: forward_n_tokens
// derives it from the executor's running position, which is zero on a cold
// prefill and the restored checkpoint position on a warm suffix-only prefill.
// Keeping the overlap calculation in this one helper prevents cold/warm paths
// from accidentally mixing prompt-absolute and suffix-relative coordinates.
constexpr QuerySpanOverlap query_span_overlap(uint32_t chunk_begin,
                                               uint32_t chunk_tokens,
                                               uint32_t query_begin,
                                               uint32_t query_end) {
    const uint32_t chunk_end = chunk_begin + chunk_tokens;
    const uint32_t lo = chunk_begin > query_begin ? chunk_begin : query_begin;
    const uint32_t hi = chunk_end < query_end ? chunk_end : query_end;
    return hi > lo ? QuerySpanOverlap{lo - chunk_begin, hi - lo}
                   : QuerySpanOverlap{};
}

// Cold full-prefill and warm checkpoint-resume regression cases. The latter is
// the LongMemEval failure mode: query [109902,109949) lies in a suffix chunk
// beginning at restored position 109888 and must capture all 47 rows.
static_assert(query_span_overlap(0, 256, 104, 151).row_offset == 104);
static_assert(query_span_overlap(0, 256, 104, 151).count == 47);
static_assert(query_span_overlap(109888, 68, 109902, 109949).row_offset == 14);
static_assert(query_span_overlap(109888, 68, 109902, 109949).count == 47);
static_assert(query_span_overlap(100, 16, 200, 220).count == 0);

class LocalKvPagePool final : public KvPhysicalPageAllocator {
public:
    LocalKvPagePool(uint32_t total_pages, uint32_t page_size,
                    const char *label)
        : total_pages_(std::max<uint32_t>(1, total_pages)),
          page_size_(std::max<uint32_t>(1, page_size)),
          label_(label ? label : "unknown"),
          in_free_(total_pages_, true) {
        free_pages_.reserve(total_pages_);
        for (uint32_t i = 0; i < total_pages_; ++i) {
            free_pages_.push_back(static_cast<int32_t>(total_pages_ - 1U - i));
        }
    }

    int32_t allocate_physical_page() override {
        if (free_pages_.empty()) {
            throw std::runtime_error(
                "local KVMem GPU page pool exhausted: pool=" + label_ +
                " free=0 total=" +
                std::to_string(total_pages_) +
                " page_size=" + std::to_string(page_size_));
        }
        const int32_t page = free_pages_.back();
        free_pages_.pop_back();
        if (page >= 0 && static_cast<uint32_t>(page) < in_free_.size()) {
            in_free_[page] = false;
        }
        ++used_pages_;
        return page;
    }

    void release_physical_pages(const std::vector<int32_t> &pages) override {
        for (int32_t page : pages) {
            if (page < 0 || static_cast<uint32_t>(page) >= total_pages_) {
                continue;
            }
            if (in_free_[page]) continue;
            in_free_[page] = true;
            free_pages_.push_back(page);
            if (used_pages_ > 0) --used_pages_;
        }
    }

    uint32_t free_pages() const override {
        return static_cast<uint32_t>(free_pages_.size());
    }

    uint32_t used_pages() const override { return used_pages_; }
    uint32_t total_pages() const override { return total_pages_; }

private:
    uint32_t total_pages_ = 0;
    uint32_t page_size_ = 0;
    std::string label_;
    std::vector<int32_t> free_pages_;
    std::vector<bool> in_free_;
    uint32_t used_pages_ = 0;
};

bool kvmem_tier_trace_enabled() {
    return env_flag_enabled("QW3_KVMEM_TIER_TRACE");
}

// Tier the MTP layer's KV through the bounded kvmem block store (default ON).
// QW3_KVMEM_MTP_TIER=0 forces the old dense full-context MTP cache under kvmem,
// for A/B comparison and rollback.
bool kvmem_mtp_tier_enabled() {
    return env_flag_enabled("QW3_KVMEM_MTP_TIER", true);
}

// Long-context MTP path (default ON with immutable K). Logical token/page
// identities remain monotonic, but every MTP Q/K is baked directly in the
// compact KVMem window frame and historical MTP K is rebuilt from an unrotated
// CPU authority. QW3_KVMEM_MTP_LOCAL_POSITIONS=0 keeps the legacy path for A/B.
bool kvmem_mtp_local_positions_enabled() {
    return env_flag_enabled("QW3_KVMEM_MTP_LOCAL_POSITIONS", true);
}

// ---- KVMem component timing (env QW3_KVMEM_TIMING; OFF by default) ---------
// Process-global wall-clock accumulators for the tier/selection components so
// the latency-breakdown harness can attribute TTFT/TBT to retrieval (selection
// scoring + pick_topk), stage-in (CPU/NVMe -> GPU), stage-out (GPU -> CPU/NVMe)
// and window assemble (re-RoPE). When enabled the GPU-async regions (retrieval
// scoring, assemble) add a device sync so their kernel time is captured; that
// perturbs throughput, so throughput must be measured WITHOUT this flag.
bool kvmem_timing_flag() {
    return env_flag_enabled("QW3_KVMEM_TIMING");
}

// Emit one detailed line per semantic re-selection. Unlike the cumulative
// QW3_KVMEM_TIMING counters, this records how much of an asynchronous prefetch
// actually ran before finish_reselect() had to wait for it.
bool kvmem_perf_trace_flag() {
    return env_flag_enabled("QW3_KVMEM_PERF_TRACE");
}

bool kvmem_measure_timing_flag() {
    return kvmem_timing_flag() || kvmem_perf_trace_flag();
}

struct KvMemTimingTotals {
    std::atomic<uint64_t> retrieval_ns{0};
    std::atomic<uint64_t> stage_in_ns{0};
    std::atomic<uint64_t> stage_out_ns{0};
    std::atomic<uint64_t> assemble_ns{0};
    std::atomic<uint64_t> assemble_pages_ns{0};
    std::atomic<uint64_t> assemble_rerope_ns{0};
    std::atomic<uint64_t> assemble_kbar_ns{0};
    std::atomic<uint32_t> retrieval_calls{0};
    std::atomic<uint32_t> stage_in_calls{0};
    std::atomic<uint32_t> stage_out_calls{0};
    std::atomic<uint32_t> assemble_calls{0};
    std::atomic<uint32_t> stage_in_blocks{0};
    std::atomic<uint32_t> stage_out_blocks{0};
};

KvMemTimingTotals &kvmem_timing_totals() {
    static KvMemTimingTotals totals;
    return totals;
}

uint64_t kvmem_steady_ns() {
    using clk = std::chrono::steady_clock;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            clk::now().time_since_epoch())
            .count());
}

bool mtp_prefix_batch_enabled() {
    return env_flag_enabled("QW3_MTP_PREFIX_BATCH", true);
}

bool mtp_paged_prefix_enabled() {
    return env_flag_enabled("QW3_MTP_PAGED_PREFIX");
}

uint32_t mtp_prefix_batch_min_tokens() {
    return env_uint32_or("QW3_MTP_PREFIX_BATCH_MIN", 32);
}

uint32_t count_standard_attention_layers(const QwenConfig &cfg,
                                         uint32_t n_layers) {
    uint32_t out = 0;
    for (uint32_t il = 0; il < n_layers; ++il) {
        if (cfg.is_standard_attention_layer(il)) ++out;
    }
    return out;
}

uint64_t estimate_kvmem_block_bytes(const QwenConfig &cfg,
                                    uint32_t standard_layers,
                                    uint32_t block_tokens) {
    const char *kv_dtype_env = std::getenv("QW3_KV_DTYPE");
    const bool kv_use_fp32 = kv_dtype_env && std::strcmp(kv_dtype_env, "fp32") == 0;
    const bool kv_use_q8 = kv_dtype_env && std::strcmp(kv_dtype_env, "q8") == 0;
    const bool kv_use_fp8 = kv_dtype_env && std::strcmp(kv_dtype_env, "fp8") == 0;
    const uint64_t elem_bytes = kv_use_fp32 ? 4ull : (kv_use_fp8 || kv_use_q8 ? 1ull : 2ull);
    const uint64_t kv_values =
        static_cast<uint64_t>(block_tokens) * cfg.n_kv_heads * cfg.head_dim;
    uint64_t bytes = static_cast<uint64_t>(standard_layers) * kv_values *
                     2ull * elem_bytes;  // K + V
    if (kv_use_q8) {
        // q8 stores one fp16 scale per row; rows are token x kv-head, for K and V.
        bytes += static_cast<uint64_t>(standard_layers) * block_tokens *
                 cfg.n_kv_heads * 2ull * sizeof(uint16_t);
    }
    return bytes;
}

} // namespace

bool QwenExecutor::kvmem_timing_enabled() { return kvmem_timing_flag(); }

QwenExecutor::KvMemTimingSnapshot QwenExecutor::kvmem_timing_snapshot() {
    const KvMemTimingTotals &t = kvmem_timing_totals();
    KvMemTimingSnapshot s;
    s.retrieval_ns = t.retrieval_ns.load(std::memory_order_relaxed);
    s.stage_in_ns = t.stage_in_ns.load(std::memory_order_relaxed);
    s.stage_out_ns = t.stage_out_ns.load(std::memory_order_relaxed);
    s.assemble_ns = t.assemble_ns.load(std::memory_order_relaxed);
    s.assemble_pages_ns = t.assemble_pages_ns.load(std::memory_order_relaxed);
    s.assemble_rerope_ns = t.assemble_rerope_ns.load(std::memory_order_relaxed);
    s.assemble_kbar_ns = t.assemble_kbar_ns.load(std::memory_order_relaxed);
    s.retrieval_calls = t.retrieval_calls.load(std::memory_order_relaxed);
    s.stage_in_calls = t.stage_in_calls.load(std::memory_order_relaxed);
    s.stage_out_calls = t.stage_out_calls.load(std::memory_order_relaxed);
    s.assemble_calls = t.assemble_calls.load(std::memory_order_relaxed);
    s.stage_in_blocks = t.stage_in_blocks.load(std::memory_order_relaxed);
    s.stage_out_blocks = t.stage_out_blocks.load(std::memory_order_relaxed);
    return s;
}

QwenExecutor::KvMemTierUsage QwenExecutor::kvmem_tier_usage() const {
    KvMemTierUsage u;
    if (!kvmem_enabled_ || !block_store_) return u;
    u.enabled = true;
    u.active = kvmem_active_;
    const KvMemStoreConfig &cfg = block_store_->config();
    u.total_blocks = block_store_->block_count();
    u.block_bytes = cfg.estimated_block_bytes;
    const uint32_t page_size = kv_pages_.page_size ? kv_pages_.page_size : 1u;
    const uint32_t pages_per_block =
        std::max<uint32_t>(1u, cfg.block_tokens / page_size);
    if (kvmem_gpu_page_pool_) {
        // Bounded pool engaged: GPU holds only the resident working set; the
        // rest has spilled to the CPU/NVMe tiers.
        u.gpu_pool = true;
        const uint64_t used_blk =
            kvmem_gpu_page_pool_->used_pages() / pages_per_block;
        const uint64_t cap_blk =
            (static_cast<uint64_t>(kvmem_gpu_page_pool_->used_pages()) +
             kvmem_gpu_page_pool_->free_pages()) / pages_per_block;
        u.gpu_used_bytes = used_blk * u.block_bytes;
        u.gpu_capacity_bytes = cap_blk * u.block_bytes;
    } else {
        // No bounded pool: every block is resident on the GPU in the normal KV
        // cache, so the GPU footprint is the whole store.
        u.gpu_used_bytes = u.total_blocks * u.block_bytes;
        u.gpu_capacity_bytes = u.gpu_used_bytes;
    }
    if (kvmem_cpu_tier_) {
        u.cpu_tier = true;
        u.cpu_spill_bytes =
            static_cast<uint64_t>(kvmem_cpu_tier_->used_slots()) * u.block_bytes;
        u.cpu_raw_k_bytes =
            kvmem_raw_k_mirror_bytes_ + kvmem_raw_mtp_k_mirror_bytes_;
        if (kvmem_sparse_cpu_tier_) {
            u.cpu_used_bytes = u.cpu_raw_k_bytes + kvmem_cpu_sparse_bytes_;
            u.cpu_capacity_bytes = kvmem_cpu_budget_bytes_;
        } else {
            u.cpu_used_bytes = u.cpu_spill_bytes;
            u.cpu_capacity_bytes =
                static_cast<uint64_t>(kvmem_cpu_tier_->slot_count()) *
                u.block_bytes;
        }
    }
    if (kvmem_nvme_tier_) {
        u.nvme_tier = true;
        u.nvme_used_bytes =
            static_cast<uint64_t>(kvmem_nvme_tier_->used_slots()) * u.block_bytes;
        u.nvme_capacity_bytes =
            static_cast<uint64_t>(kvmem_nvme_tier_->slot_count()) * u.block_bytes;
    }
    return u;
}

void QwenExecutor::kvmem_timing_emit_delta(const char *tag,
                                           const KvMemTimingSnapshot &base) {
    const KvMemTimingSnapshot now = kvmem_timing_snapshot();
    std::fprintf(
        stderr,
        "[kvmem-timing] %s retrieval_ms=%.3f stage_in_ms=%.3f "
        "stage_out_ms=%.3f assemble_ms=%.3f (pages=%.3f rerope=%.3f kbar=%.3f) "
        "| retrieval=%u stage_in=%u(%ublk) "
        "stage_out=%u(%ublk) assemble=%u\n",
        tag ? tag : "",
        (now.retrieval_ns - base.retrieval_ns) / 1e6,
        (now.stage_in_ns - base.stage_in_ns) / 1e6,
        (now.stage_out_ns - base.stage_out_ns) / 1e6,
        (now.assemble_ns - base.assemble_ns) / 1e6,
        (now.assemble_pages_ns - base.assemble_pages_ns) / 1e6,
        (now.assemble_rerope_ns - base.assemble_rerope_ns) / 1e6,
        (now.assemble_kbar_ns - base.assemble_kbar_ns) / 1e6,
        now.retrieval_calls - base.retrieval_calls,
        now.stage_in_calls - base.stage_in_calls,
        now.stage_in_blocks - base.stage_in_blocks,
        now.stage_out_calls - base.stage_out_calls,
        now.stage_out_blocks - base.stage_out_blocks,
        now.assemble_calls - base.assemble_calls);
}

QwenExecutor::QwenExecutor(const QwenNativeModel &model,
                           const QwenWeights &weights,
                           DeviceBackend &backend,
                           uint32_t kv_ctx_size,
                           KvPhysicalPageAllocator *kv_page_allocator,
                           KvCacheStorage *external_kv_cache,
                           KvPhysicalPageAllocator *mtp_kv_page_allocator,
                           KvCacheStorage *external_mtp_kv_cache)
    : model_(model), weights_(weights), backend_(backend),
      external_kv_cache_(external_kv_cache),
      external_mtp_kv_cache_(external_mtp_kv_cache),
      kv_ctx_size_(kv_ctx_size) {
    kv_pages_.configure(kv_ctx_size_, kv_page_allocator);
    mtp_kv_pages_.configure(kv_ctx_size_, mtp_kv_page_allocator);
}

QwenExecutor::~QwenExecutor() {
    try {
        kvmem_finish_proactive_d2h(/*wait_all=*/true);
        kvmem_reap_pending_writes(/*wait_all=*/true);
    } catch (const std::exception &e) {
        std::fprintf(stderr,
                     "[kvmem-io] pending write failed during destruction: %s\n",
                     e.what());
    }
    if (kvmem_attn_trace_file_) {
        std::fclose(kvmem_attn_trace_file_);
        kvmem_attn_trace_file_ = nullptr;
    }
    if (global_attn_trace_file_) {
        std::fclose(global_attn_trace_file_);
        global_attn_trace_file_ = nullptr;
    }
    kv_pages_.reset();
    mtp_kv_pages_.reset();
    // Return the pinned CPU-tier buffer to the shared pool (if borrowed) so the
    // next executor reuses it instead of paying cudaHostAlloc again.
    if (host_tier_pool_ && kvmem_cpu_bytes_) {
        host_tier_pool_->release(std::move(kvmem_cpu_bytes_));
    }
}

QwenExecutor::DecodeStateView QwenExecutor::decode_state_view() const {
    DecodeStateView view;
    view.position = position_;
    view.kv_ctx_size = kv_ctx_size_;
    view.kv_page_size = kv_pages_.page_size;
    view.kv_page_count = kv_pages_.count();
    view.kv_page_indices_host = kv_pages_.host_indices();
    view.kv_page_indices_device = kv_pages_.device_pages.get();
    view.k_cache = &k_cache_;
    view.v_cache = &v_cache_;
    view.k_cache_external = external_kv_cache_ ? &external_kv_cache_->k_cache : nullptr;
    view.v_cache_external = external_kv_cache_ ? &external_kv_cache_->v_cache : nullptr;
    view.recurrent_states = &recurrent_states_;
    view.conv_states = &conv_states_;
    view.hidden = h_.get();
    return view;
}

QwenExecutor::MutableDecodeStateView QwenExecutor::mutable_decode_state_view() {
    MutableDecodeStateView view;
    view.position = position_;
    view.kv_ctx_size = kv_ctx_size_;
    view.kv_page_size = kv_pages_.page_size;
    view.kv_page_count = kv_pages_.count();
    view.kv_page_indices_host = kv_pages_.host_indices();
    view.kv_page_indices_device = kv_pages_.device_pages.get();
    view.k_cache = &k_cache_;
    view.v_cache = &v_cache_;
    view.k_cache_external = external_kv_cache_ ? &external_kv_cache_->k_cache : nullptr;
    view.v_cache_external = external_kv_cache_ ? &external_kv_cache_->v_cache : nullptr;
    view.recurrent_states = &recurrent_states_;
    view.conv_states = &conv_states_;
    view.hidden = h_.get();
    return view;
}

QwenExecutor::MtpPrefixStateView QwenExecutor::mtp_prefix_state_view() {
    ensure_mtp_scratch();
    MtpPrefixStateView view;
    DeviceTensor *mtp_k_cache =
        external_mtp_kv_cache_ &&
                !external_mtp_kv_cache_->k_cache.empty()
            ? external_mtp_kv_cache_->k_cache.front()
            : mtp_k_cache_.get();
    DeviceTensor *mtp_v_cache =
        external_mtp_kv_cache_ &&
                !external_mtp_kv_cache_->v_cache.empty()
            ? external_mtp_kv_cache_->v_cache.front()
            : mtp_v_cache_.get();
    view.ready = mtp_scratch_ready_ && mtp_k_cache && mtp_v_cache &&
                 mtp_prefix_h_ && h_;
    view.prefix_len = mtp_prefix_len_;
    view.ctx_size = kv_ctx_size_;
    view.page_size = mtp_kv_pages_.page_size;
    view.page_count = mtp_kv_pages_.count();
    view.page_indices_host = mtp_kv_pages_.host_indices();
    view.page_indices_device = mtp_kv_pages_.device_pages.get();
    view.k_cache = mtp_k_cache;
    view.v_cache = mtp_v_cache;
    view.prefix_hidden = mtp_prefix_h_.get();
    view.current_hidden = h_.get();
    view.draft_hidden = mtp_h_.get();
    return view;
}

void QwenExecutor::prepare_decode_token_pages(uint32_t count) {
    ensure_kv_pages(position_, count);
}

void QwenExecutor::prepare_runtime_state() {
    ensure_scratch();
}

void QwenExecutor::prepare_kv_pages(uint32_t logical_pos, uint32_t count) {
    ensure_kv_pages(logical_pos, count);
}

void QwenExecutor::prepare_mtp_prefix_pages(uint32_t logical_pos,
                                            uint32_t count) {
    ensure_mtp_scratch();
    mtp_kv_pages_.ensure_pages(backend_, kv_ctx_size_, logical_pos, count);
    if (external_mtp_kv_cache_) {
        mtp_kv_pages_.validate_physical_capacity(
            external_mtp_kv_cache_->physical_slots, "external MTP");
    }
}

void QwenExecutor::set_mtp_prefix_len(uint32_t prefix_len) {
    mtp_prefix_len_ = std::min<uint32_t>(prefix_len, kv_ctx_size_);
}

DeviceTensor &QwenExecutor::k_cache(uint32_t layer) {
    if (external_kv_cache_) {
        if (layer >= external_kv_cache_->k_cache.size() ||
            external_kv_cache_->k_cache[layer] == nullptr) {
            throw std::runtime_error("external K cache missing for layer " +
                                     std::to_string(layer));
        }
        return *external_kv_cache_->k_cache[layer];
    }
    if (layer >= k_cache_.size() || !k_cache_[layer]) {
        throw std::runtime_error("K cache missing for layer " +
                                 std::to_string(layer));
    }
    return *k_cache_[layer];
}

DeviceTensor &QwenExecutor::attention_k_cache(uint32_t layer) {
    return k_cache(layer);
}

DeviceTensor &QwenExecutor::v_cache(uint32_t layer) {
    if (external_kv_cache_) {
        if (layer >= external_kv_cache_->v_cache.size() ||
            external_kv_cache_->v_cache[layer] == nullptr) {
            throw std::runtime_error("external V cache missing for layer " +
                                     std::to_string(layer));
        }
        return *external_kv_cache_->v_cache[layer];
    }
    if (layer >= v_cache_.size() || !v_cache_[layer]) {
        throw std::runtime_error("V cache missing for layer " +
                                 std::to_string(layer));
    }
    return *v_cache_[layer];
}

DeviceTensor &QwenExecutor::mtp_k_cache() {
    if (external_mtp_kv_cache_) {
        if (external_mtp_kv_cache_->k_cache.empty() ||
            external_mtp_kv_cache_->k_cache.front() == nullptr) {
            throw std::runtime_error("external MTP K cache missing");
        }
        return *external_mtp_kv_cache_->k_cache.front();
    }
    if (!mtp_k_cache_) {
        throw std::runtime_error("MTP K cache missing");
    }
    return *mtp_k_cache_;
}

DeviceTensor &QwenExecutor::mtp_v_cache() {
    if (external_mtp_kv_cache_) {
        if (external_mtp_kv_cache_->v_cache.empty() ||
            external_mtp_kv_cache_->v_cache.front() == nullptr) {
            throw std::runtime_error("external MTP V cache missing");
        }
        return *external_mtp_kv_cache_->v_cache.front();
    }
    if (!mtp_v_cache_) {
        throw std::runtime_error("MTP V cache missing");
    }
    return *mtp_v_cache_;
}

QwenExecutor::KvStateSnapshot QwenExecutor::kv_state_snapshot() const {
    KvStateSnapshot snapshot;
    snapshot.seq_len = position_;
    snapshot.ctx_size = kv_ctx_size_;
    snapshot.page_size = kv_pages_.page_size;
    snapshot.logical_pages = kv_pages_.count();
    snapshot.physical_pages = kv_pages_.pages;
    return snapshot;
}

void QwenExecutor::reset_state() {
    const uint64_t cold_reset_start_ns = kvmem_steady_ns();
    kvmem_finish_proactive_d2h(/*wait_all=*/true);
    const uint64_t cold_reset_after_proactive_ns = kvmem_steady_ns();
    kvmem_reap_pending_writes(/*wait_all=*/true);
    const uint64_t cold_reset_after_pending_ns = kvmem_steady_ns();
    for (auto &s : recurrent_states_) {
        if (s) (void) backend_.zero_tensor(*s);
    }
    for (auto &s : conv_states_) {
        if (s) (void) backend_.zero_tensor(*s);
    }
    const uint64_t cold_reset_after_state_zero_ns = kvmem_steady_ns();
    // KV caches stay allocated; just reset the position so the next forward
    // overwrites slot 0 (the seq_len passed to attention_decode is position+1).
    position_ = 0;
    kv_pages_.reset();
    mtp_kv_pages_.reset();
    mtp_prefix_len_ = 0;
    decode_graph_warmup_pending_ = true;
    const uint64_t cold_reset_after_page_reset_ns = kvmem_steady_ns();
    // Block-sparse runtime state is per-session: drop the working set + window
    // table. The block store itself (configured selection params) is kept; a
    // fresh register_append sequence rebuilds the block table for the new run.
    kvmem_active_ = false;
    kvmem_registered_pos_ = 0;
    kvmem_query_replay_active_ = false;
    kvmem_query_prefetch_active_ = false;
    kvmem_query_prefetch_blocks_.clear();
    kvmem_query_prefetch_start_ns_ = 0;
    kvmem_keep_selected_prefill_ = false;
    kvmem_defer_prefill_pressure_ = false;
    kvmem_deferred_prefill_tokens_ = 0;
    last_forward_logical_base_ = 0;
    last_forward_rope_base_ = 0;
    last_forward_rows_ = 0;
    window_pages_host_.clear();
    window_page_count_ = 0;
    mtp_window_pages_host_.clear();
    mtp_window_page_count_ = 0;
    window_query_pos_ = 0;
    const uint64_t cold_reset_raw_bytes =
        kvmem_raw_k_mirror_bytes_ + kvmem_raw_mtp_k_mirror_bytes_;
    const uint64_t cold_reset_sparse_bytes = kvmem_cpu_sparse_bytes_;
    // Opt3 keeps the bounded host allocation topology warm across independent
    // requests. The contents are never reused as cache authority: clear() and
    // the validity maps below invalidate every block/token. Only the backing
    // allocations survive, capped by --kvmem-cpu-gb. This avoids repeatedly
    // freeing, trimming, faulting, and rebuilding tens of GiB on the request
    // thread. Lower optimization levels retain the compatibility behavior, and
    // deployments that prefer minimum idle RSS can explicitly disable it.
    const bool retain_cold_host =
        block_store_ && kvmem_sparse_cpu_tier_ &&
        block_store_->config().optimization_level >=
            KvMemOptimizationLevel::Opt3 &&
        env_flag_enabled("QW3_KVMEM_RETAIN_COLD_HOST", true);
    if (kvmem_cpu_tier_) kvmem_cpu_tier_->clear();
    if (kvmem_sparse_cpu_tier_) {
        for (auto &slab : kvmem_cpu_sparse_slabs_) {
            slab.live_slots = 0;
            if (!retain_cold_host) slab.data.reset();
        }
        std::fill(kvmem_cpu_sparse_slot_live_.begin(),
                  kvmem_cpu_sparse_slot_live_.end(), uint8_t{0});
        if (!retain_cold_host) kvmem_cpu_sparse_bytes_ = 0;
    }
    if (kvmem_nvme_tier_) kvmem_nvme_tier_->clear();
    const uint64_t cold_reset_after_tier_clear_ns = kvmem_steady_ns();
    kvmem_stage_pinned_.reset();
    kvmem_prefetch_ = KvMemPrefetchState{};
    kvmem_last_prefetch_perf_ = KvMemPrefetchPerf{};
    kvmem_last_stage_out_perf_ = KvMemStageOutPerf{};
    kvmem_reselect_perf_ = KvMemReselectPerf{};
    kvmem_writeback_next_block_ = 0;
    kvmem_writeback_d2h_bytes_ = 0;
    kvmem_writeback_d2h_wait_ns_ = 0;
    kvmem_writeback_d2h_batches_ = 0;
    kvmem_writeback_blocks_ = 0;
    kvmem_pending_reselect_ = false;
    kvmem_pending_plan_ = KvMemPlan{};
    // reset_state() is the cold-request boundary. Warm prefix continuation uses
    // restore_state()+kvmem_truncate_to() and deliberately does not come through
    // here, so no raw-K authority belongs to the next request. Opt3 may retain
    // the bounded allocations as an invalid buffer pool; lower levels release
    // their physical contents and allocate again on demand.
    if (!retain_cold_host) kvmem_truncate_raw_k(0);
    std::fill(kvmem_raw_k_valid_tokens_.begin(),
              kvmem_raw_k_valid_tokens_.end(), uint8_t{0});
    std::fill(kvmem_raw_mtp_k_valid_tokens_.begin(),
              kvmem_raw_mtp_k_valid_tokens_.end(), uint8_t{0});
#if defined(__GLIBC__)
    // glibc can adapt its mmap threshold upward after these large, repeated
    // allocations. Subsequent raw-K chunks / sparse spill blocks may then come
    // from heap arenas; delete[] makes them reusable but does not necessarily
    // return their pages to the OS. At a cold request boundary no KVMem host
    // allocation is live, so trim those free arena pages now. This is not used
    // by warm-prefix restore/truncate and therefore cannot discard reusable
    // prefix state.
    const int cold_reset_trimmed =
        !retain_cold_host &&
        (cold_reset_raw_bytes != 0 || cold_reset_sparse_bytes != 0)
            ? ::malloc_trim(0)
            : 0;
#else
    const int cold_reset_trimmed = -1;
#endif
    const uint64_t cold_reset_after_host_release_ns = kvmem_steady_ns();
    kvmem_raw_decode_block_start_ = -1;
    kvmem_raw_decode_first_row_ = 0;
    kvmem_raw_decode_rows_ = 0;
    // Cumulative-attention selection signal: drop the live interval; buffers stay
    // allocated (reused next session). bs_score_layer_ is model-fixed, keep it.
    bs_score_ready_ = false;
    bs_window_blocks_ = 0;
    bs_window_block_ids_.clear();
    bs_win_base_host_.clear();
    bs_blk_tokens_host_.clear();
    // Global content-frame retrieval index is per-session: drop it so the next
    // run rebuilds from its own pristine post-prefill cache. Device buffers stay
    // allocated (reused next session, regrown by block capacity).
    g_content_ready_ = false;
    g_query_ready_ = false;
    g_indexed_blocks_ = 0;
    g_orig_base_host_.clear();
    g_blk_tokens_host_.clear();
    // Query-conditioned span is per-request: clear it so a request WITHOUT a
    // span (or with query-conditioning off) never inherits a prior request's
    // span. The device buffer stays allocated (reused, regrown by set_query_span).
    kvmem_drain_query_capture();
    kvmem_query_begin_ = 0;
    kvmem_query_end_ = 0;
    g_query_multi_count_ = 0;
    g_query_multi_ready_ = false;
    kvmem_qc_capture_active_ = false;
    // Per-layer multi-layer index + incremental-capture progress are per-request:
    // drop them so a fresh request rebuilds its own full-coverage index (#91).
    g_kbar_multi_ready_ = false;
    g_kbar_multi_blocks_ = 0;
    kvmem_qc_total_blocks_ = 0;
    kvmem_qc_prompt_tokens_ = 0;
    kvmem_qc_captured_blocks_ = 0;
    kvmem_qc_captured_tokens_ = 0;
    // DeltaNet-state retrieval capture is per-request (device buffers stay
    // allocated for reuse; regrown by set_query_span as the block count grows).
    kvmem_dn_ready_ = false;
    kvmem_dn_qcount_ = 0;
    // A full reset is a cold start: no prefix survives, so clear the fixed stride
    // and the session-continuation resume base. The next kvmem_set_query_span then
    // zeroes the whole content index (resume_base == 0) and re-pins the stride.
    kvmem_qc_layer_stride_blocks_ = 0;
    kvmem_qc_resume_base_tokens_ = 0;
    // Clean-query prefill (task #50): drop the decode-window pin so a request that
    // does NOT use clean-query never inherits a stale pin. Do NOT clear
    // g_query_multi_clean_count_ here: PASS A stashes the clean query and THEN calls
    // reset_state before PASS B restores it, so clearing the count would lose the
    // stash. It is instead overwritten by the next PASS-A stash and only ever read
    // after a stash within the same request.
    kvmem_qc_pin_from_block_ = 0xffffffffu;
    kvmem_oracle_token_spans_.clear();
    kvmem_oracle_only_ = false;
    kvmem_trace_event_index_ = 0;
    kvmem_trace_event_count_ = 0;
    if (block_store_) {
        *block_store_ = KvMemStore(block_store_->config());
    }
    kvmem_attn_trace_seen_tokens_ = 0;
    kvmem_attn_trace_sample_ = 0;
    global_attn_trace_seen_tokens_ = 0;
    global_attn_trace_sample_ = 0;
    const uint64_t cold_reset_end_ns = kvmem_steady_ns();
    if ((cold_reset_raw_bytes != 0 || cold_reset_sparse_bytes != 0) &&
        (kvmem_tier_trace_enabled() || kvmem_timing_flag())) {
        const auto ms = [](uint64_t begin, uint64_t end) {
            return (end - begin) / 1.0e6;
        };
        std::fprintf(
            stderr,
            "[kvmem-tier] cold_reset_host_release raw_bytes=%llu "
            "sparse_bytes=%llu retain=%d allocator_trim=%d total_ms=%.3f "
            "proactive_ms=%.3f pending_ms=%.3f state_zero_ms=%.3f "
            "page_reset_ms=%.3f tier_clear_ms=%.3f host_release_ms=%.3f "
            "metadata_ms=%.3f\n",
            static_cast<unsigned long long>(cold_reset_raw_bytes),
            static_cast<unsigned long long>(cold_reset_sparse_bytes),
            retain_cold_host ? 1 : 0,
            cold_reset_trimmed,
            ms(cold_reset_start_ns, cold_reset_end_ns),
            ms(cold_reset_start_ns, cold_reset_after_proactive_ns),
            ms(cold_reset_after_proactive_ns, cold_reset_after_pending_ns),
            ms(cold_reset_after_pending_ns,
               cold_reset_after_state_zero_ns),
            ms(cold_reset_after_state_zero_ns,
               cold_reset_after_page_reset_ns),
            ms(cold_reset_after_page_reset_ns,
               cold_reset_after_tier_clear_ns),
            ms(cold_reset_after_tier_clear_ns,
               cold_reset_after_host_release_ns),
            ms(cold_reset_after_host_release_ns, cold_reset_end_ns));
    }
}

void QwenExecutor::KvPageTable::configure(uint32_t ctx_size,
                                          KvPhysicalPageAllocator *page_allocator) {
    page_size = std::max<uint32_t>(1, env_uint32_or("QW3_PAGED_KV_PAGE_SIZE", 16));
    max_pages = (ctx_size + page_size - 1) / page_size;
    allocator = page_allocator;
    alloc_mode = env_lower_ascii(env_value("QW3_PAGED_KV_ALLOC"));
    if (alloc_mode.empty()) alloc_mode = "identity";
    if (alloc_mode != "identity" &&
        alloc_mode != "reverse" &&
        alloc_mode != "evens-first") {
        throw std::runtime_error("invalid QW3_PAGED_KV_ALLOC: " + alloc_mode +
                                 " (want identity|reverse|evens-first)");
    }
    reset();
}

void QwenExecutor::KvPageTable::set_allocator(
        KvPhysicalPageAllocator *page_allocator) {
    if (!pages.empty()) {
        throw std::runtime_error(
            "cannot change KV page allocator after pages have been allocated");
    }
    allocator = page_allocator;
    device_synced = 0;
}

void QwenExecutor::KvPageTable::reset() {
    if (allocator && !pages.empty()) {
        // Only release pages this table actually owns. Borrowed (prefix-cache)
        // pages are pinned elsewhere and must never go back on the free stack.
        std::vector<int32_t> to_release;
        to_release.reserve(pages.size());
        for (size_t i = 0; i < pages.size(); ++i) {
            if (i >= owned.size() || owned[i]) to_release.push_back(pages[i]);
        }
        if (!to_release.empty()) allocator->release_physical_pages(to_release);
    }
    pages.clear();
    owned.clear();
    device_synced = 0;
}

void QwenExecutor::KvPageTable::ensure_pages(DeviceBackend &backend,
                                             uint32_t ctx_size,
                                             uint32_t logical_pos,
                                             uint32_t count) {
    if (count == 0) return;
    const uint64_t end_pos = static_cast<uint64_t>(logical_pos) + count;
    if (end_pos > ctx_size) {
        throw std::runtime_error("KV cache full: increase --ctx (current=" +
                                 std::to_string(ctx_size) + ")");
    }
    const uint32_t need_pages =
        static_cast<uint32_t>((end_pos + page_size - 1) / page_size);
    if (need_pages > max_pages) {
        throw std::runtime_error("KV page table full: increase --ctx (current=" +
                                 std::to_string(ctx_size) + ")");
    }
    while (pages.size() < need_pages) {
        const uint32_t logical_page = static_cast<uint32_t>(pages.size());
        const int32_t physical_page =
            allocator ? allocator->allocate_physical_page()
                      : allocate_physical_page(logical_page);
        if (physical_page < 0 ||
            static_cast<uint32_t>(physical_page) >=
                (allocator ? allocator->total_pages() : max_pages)) {
            if (allocator) {
                allocator->release_physical_pages(std::vector<int32_t>{physical_page});
            }
            throw std::runtime_error(
                "KV physical page allocation returned out-of-range page " +
                std::to_string(physical_page) + " for max_pages=" +
                std::to_string(max_pages));
        }
        pages.push_back(physical_page);
        owned.push_back(true);
    }
    if (!device_pages) {
        device_pages = backend.tensor_i32(std::max<uint32_t>(max_pages, 1),
                                          "kv_page_indices");
        device_synced = 0;
    }
    const uint32_t have_pages = static_cast<uint32_t>(pages.size());
    if (device_synced < have_pages) {
        const uint32_t start = device_synced;
        const uint32_t count_new = have_pages - start;
        require_status(backend.copy_i32_from_host(
            *device_pages, start, pages.data() + start, count_new));
        device_synced = have_pages;
    }
}

void QwenExecutor::KvPageTable::validate_physical_capacity(
        uint64_t physical_slots, const char *label) const {
    if (page_size == 0) {
        throw std::runtime_error(std::string(label) +
                                 " KV page size is zero");
    }
    const uint64_t physical_pages =
        (physical_slots + page_size - 1) / page_size;
    for (int32_t page : pages) {
        if (page < 0) continue;
        if (static_cast<uint64_t>(page) >= physical_pages) {
            throw std::runtime_error(
                std::string(label) +
                " KV physical page out of cache bounds: page=" +
                std::to_string(page) +
                " physical_pages=" + std::to_string(physical_pages) +
                " physical_slots=" + std::to_string(physical_slots) +
                " page_size=" + std::to_string(page_size));
        }
    }
}

void QwenExecutor::KvPageTable::truncate_to_logical_pages(uint32_t logical_pages) {
    if (logical_pages >= pages.size()) return;
    std::vector<int32_t> released;
    released.reserve(pages.size() - logical_pages);
    for (size_t i = logical_pages; i < pages.size(); ++i) {
        // Borrowed (prefix-cache) pages are pinned elsewhere; never release.
        if (i >= owned.size() || owned[i]) released.push_back(pages[i]);
    }
    pages.resize(logical_pages);
    if (owned.size() > logical_pages) owned.resize(logical_pages);
    device_synced = std::min<uint32_t>(device_synced, logical_pages);
    if (allocator && !released.empty()) {
        allocator->release_physical_pages(released);
    }
}

bool QwenExecutor::KvPageTable::logical_page_resident(
        uint32_t logical_page) const {
    return logical_page < pages.size() && pages[logical_page] >= 0;
}

int32_t QwenExecutor::KvPageTable::ensure_logical_page_resident(
        DeviceBackend &backend, uint32_t logical_page) {
    if (logical_page >= max_pages) {
        throw std::runtime_error("KV logical page exceeds page capacity");
    }
    while (pages.size() <= logical_page) {
        const uint32_t lp = static_cast<uint32_t>(pages.size());
        const int32_t physical_page =
            allocator ? allocator->allocate_physical_page()
                      : allocate_physical_page(lp);
        pages.push_back(physical_page);
        owned.push_back(true);
    }
    if (pages[logical_page] < 0) {
        const int32_t physical_page =
            allocator ? allocator->allocate_physical_page()
                      : allocate_physical_page(logical_page);
        pages[logical_page] = physical_page;
        if (owned.size() < pages.size()) owned.resize(pages.size(), true);
        owned[logical_page] = true;
    }
    if (!device_pages) {
        device_pages = backend.tensor_i32(std::max<uint32_t>(max_pages, 1),
                                          "kv_page_indices");
    }
    require_status(backend.copy_i32_from_host(
        *device_pages, logical_page, &pages[logical_page], 1));
    device_synced = std::max<uint32_t>(device_synced, logical_page + 1);
    return pages[logical_page];
}

void QwenExecutor::KvPageTable::ensure_logical_page_ranges_resident(
        DeviceBackend &backend,
        const std::vector<std::pair<uint32_t, uint32_t>> &ranges) {
    if (ranges.empty()) return;
    uint32_t upload_begin = max_pages;
    uint32_t upload_end = 0;
    for (const auto &[logical_start, count] : ranges) {
        if (count == 0) continue;
        const uint64_t end64 =
            static_cast<uint64_t>(logical_start) + count;
        if (logical_start >= max_pages || end64 > max_pages) {
            throw std::runtime_error(
                "KV logical page range exceeds page capacity");
        }
        const uint32_t end = static_cast<uint32_t>(end64);
        while (pages.size() < end) {
            const uint32_t lp = static_cast<uint32_t>(pages.size());
            const int32_t physical_page =
                allocator ? allocator->allocate_physical_page()
                          : allocate_physical_page(lp);
            pages.push_back(physical_page);
            owned.push_back(true);
        }
        for (uint32_t lp = logical_start; lp < end; ++lp) {
            if (pages[lp] >= 0) continue;
            const int32_t physical_page =
                allocator ? allocator->allocate_physical_page()
                          : allocate_physical_page(lp);
            pages[lp] = physical_page;
            if (owned.size() < pages.size()) {
                owned.resize(pages.size(), true);
            }
            owned[lp] = true;
        }
        upload_begin = std::min(upload_begin, logical_start);
        upload_end = std::max(upload_end, end);
    }
    if (upload_begin >= upload_end) return;
    // If the logical vector grew past the previously published prefix, include
    // that gap too; advancing device_synced across an uninitialized gap would
    // make a later ensure_pages() incorrectly treat it as resident on device.
    upload_begin = std::min(upload_begin, device_synced);
    if (!device_pages) {
        device_pages = backend.tensor_i32(
            std::max<uint32_t>(max_pages, 1), "kv_page_indices");
    }
    require_status(backend.copy_i32_from_host(
        *device_pages, upload_begin, pages.data() + upload_begin,
        upload_end - upload_begin));
    device_synced = std::max(device_synced, upload_end);
}

void QwenExecutor::KvPageTable::release_logical_pages(
        DeviceBackend &backend, uint32_t logical_start, uint32_t count) {
    if (count == 0 || logical_start >= pages.size()) return;
    const uint32_t end = std::min<uint32_t>(
        static_cast<uint32_t>(pages.size()), logical_start + count);
    std::vector<int32_t> released;
    released.reserve(end - logical_start);
    for (uint32_t lp = logical_start; lp < end; ++lp) {
        if (pages[lp] < 0) continue;
        if (lp >= owned.size() || owned[lp]) released.push_back(pages[lp]);
        pages[lp] = -1;
        if (lp < owned.size()) owned[lp] = false;
    }
    if (allocator && !released.empty()) {
        allocator->release_physical_pages(released);
    }
    if (!device_pages) return;
    require_status(backend.copy_i32_from_host(
        *device_pages, logical_start, pages.data() + logical_start,
        end - logical_start));
}

void QwenExecutor::KvPageTable::release_logical_page_ranges(
        DeviceBackend &backend,
        const std::vector<std::pair<uint32_t, uint32_t>> &ranges) {
    if (ranges.empty() || pages.empty()) return;
    uint32_t upload_begin = static_cast<uint32_t>(pages.size());
    uint32_t upload_end = 0;
    std::vector<int32_t> released;
    for (const auto &[logical_start, count] : ranges) {
        if (count == 0 || logical_start >= pages.size()) continue;
        const uint32_t end = std::min<uint32_t>(
            static_cast<uint32_t>(pages.size()), logical_start + count);
        upload_begin = std::min(upload_begin, logical_start);
        upload_end = std::max(upload_end, end);
        for (uint32_t lp = logical_start; lp < end; ++lp) {
            if (pages[lp] < 0) continue;
            if (lp >= owned.size() || owned[lp]) {
                released.push_back(pages[lp]);
            }
            pages[lp] = -1;
            if (lp < owned.size()) owned[lp] = false;
        }
    }
    if (allocator && !released.empty()) {
        allocator->release_physical_pages(released);
    }
    if (!device_pages || upload_begin >= upload_end) return;
    require_status(backend.copy_i32_from_host(
        *device_pages, upload_begin, pages.data() + upload_begin,
        upload_end - upload_begin));
}

void QwenExecutor::KvPageTable::adopt_shared_pages(DeviceBackend &backend,
                                                  const std::vector<int32_t> &shared) {
    if (!pages.empty()) {
        throw std::runtime_error(
            "adopt_shared_pages requires a freshly-reset KV page table");
    }
    if (shared.empty()) return;
    const uint32_t need = static_cast<uint32_t>(shared.size());
    if (need > max_pages) {
        throw std::runtime_error(
            "adopt_shared_pages: shared prefix exceeds KV page capacity");
    }
    pages = shared;
    // Borrowed pages: this table never releases them (pinned by the cache).
    owned.assign(pages.size(), false);
    if (!device_pages) {
        device_pages = backend.tensor_i32(std::max<uint32_t>(max_pages, 1),
                                          "kv_page_indices");
    }
    require_status(backend.copy_i32_from_host(*device_pages, 0, pages.data(),
                                              need));
    device_synced = need;
}

std::vector<int32_t> QwenExecutor::KvPageTable::detach_pages_from(uint32_t logical_start) {
    std::vector<int32_t> detached;
    if (logical_start >= pages.size()) return detached;
    detached.assign(pages.begin() + static_cast<std::ptrdiff_t>(logical_start),
                    pages.end());
    // Drop the detached logical range WITHOUT releasing the physical pages:
    // ownership is transferred to the caller (prefix-cache entry).
    pages.resize(logical_start);
    if (owned.size() > logical_start) owned.resize(logical_start);
    device_synced = std::min<uint32_t>(device_synced, logical_start);
    return detached;
}

int32_t QwenExecutor::KvPageTable::allocate_physical_page(uint32_t logical_page) const {
    if (logical_page >= max_pages) {
        throw std::runtime_error("KV physical page allocation exceeded page capacity");
    }
    if (alloc_mode == "identity") {
        return static_cast<int32_t>(logical_page);
    }
    if (alloc_mode == "reverse") {
        return static_cast<int32_t>(max_pages - 1U - logical_page);
    }
    if (alloc_mode == "evens-first") {
        const uint32_t even_count = (max_pages + 1U) / 2U;
        const uint32_t physical_page =
            logical_page < even_count
                ? logical_page * 2U
                : (logical_page - even_count) * 2U + 1U;
        if (physical_page >= max_pages) {
            throw std::runtime_error("evens-first KV page allocator produced an invalid page");
        }
        return static_cast<int32_t>(physical_page);
    }
    throw std::runtime_error("invalid QW3_PAGED_KV_ALLOC: " + alloc_mode);
}

uint64_t QwenExecutor::KvPageTable::physical_slots() const {
    return static_cast<uint64_t>(std::max<uint32_t>(max_pages, 1)) * page_size;
}

void QwenExecutor::ensure_kv_pages(uint32_t logical_pos, uint32_t count) {
    kv_pages_.ensure_pages(backend_, kv_ctx_size_, logical_pos, count);
    if (external_kv_cache_) {
        kv_pages_.validate_physical_capacity(external_kv_cache_->physical_slots,
                                             "external");
    }
}

void QwenExecutor::begin_record_timing(bool enabled) const {
    trace_last_seconds_ = 0.0;
    if (!enabled) return;
    require_status(backend_.synchronize());
    trace_last_seconds_ = steady_seconds();
}

void QwenExecutor::record(NativeExecutorReport &report, const std::string &op) const {
    if (trace_last_seconds_ > 0.0) {
        require_status(backend_.synchronize());
        const double now = steady_seconds();
        report.elapsed_us.push_back((now - trace_last_seconds_) * 1.0e6);
        trace_last_seconds_ = now;
    }
    report.ops_executed++;
    report.executed.push_back(op);
}

void QwenExecutor::ensure_scratch() {
    if (scratch_ready_) return;
    const QwenConfig &cfg = model_.config();

    // Walk layer shapes once to find the largest per-layer dims so scratch
    // buffers can be sized to the worst case and reused across all layers.
    uint64_t max_ffn = 0;
    uint64_t max_q = 0;
    uint64_t max_k = 0;
    uint64_t max_v = 0;
    uint64_t max_recurrent_qkv = 0;
    uint64_t max_recurrent_value = 0;
    for (uint32_t i = 0; i < weights_.n_layers(); ++i) {
        const QwenLayerWeights &l = weights_.layer(i);
        if (l.ffn_dim > max_ffn) max_ffn = l.ffn_dim;
        if (l.q_rows > max_q) max_q = l.q_rows;
        if (l.k_rows > max_k) max_k = l.k_rows;
        if (l.v_rows > max_v) max_v = l.v_rows;
        if (l.recurrent_qkv_dim > max_recurrent_qkv) max_recurrent_qkv = l.recurrent_qkv_dim;
        if (l.recurrent_value_dim > max_recurrent_value) max_recurrent_value = l.recurrent_value_dim;
    }
    if (const QwenMtpWeights *mtp = weights_.mtp()) {
        const QwenLayerWeights &l = mtp->layer;
        if (l.ffn_dim > max_ffn) max_ffn = l.ffn_dim;
        if (l.q_rows > max_q) max_q = l.q_rows;
        if (l.k_rows > max_k) max_k = l.k_rows;
        if (l.v_rows > max_v) max_v = l.v_rows;
    }

    h_ = backend_.tensor_f32(cfg.n_embd, "h");
    norm_ = backend_.tensor_f32(cfg.n_embd, "norm");
    attn_out_ = backend_.tensor_f32(cfg.n_embd, "attn_out");
    ffn_gate_ = backend_.tensor_f32(std::max<uint64_t>(max_ffn, 1), "ffn_gate");
    ffn_up_ = backend_.tensor_f32(std::max<uint64_t>(max_ffn, 1), "ffn_up");
    ffn_mid_ = backend_.tensor_f32(std::max<uint64_t>(max_ffn, 1), "ffn_mid");
    ffn_out_ = backend_.tensor_f32(cfg.n_embd, "ffn_out");
    if (max_recurrent_qkv > 0) proj_ = backend_.tensor_f32(max_recurrent_qkv, "recurrent_proj");
    if (max_recurrent_value > 0) gate_proj_ = backend_.tensor_f32(max_recurrent_value, "recurrent_gate");
    if (max_recurrent_value > 0) core_ = backend_.tensor_f32(max_recurrent_value, "recurrent_core");
    if (max_q > 0) q_ = backend_.tensor_f32(max_q, "attn_q");
    if (max_k > 0) k_ = backend_.tensor_f32(max_k, "attn_k");
    if (max_v > 0) v_ = backend_.tensor_f32(max_v, "attn_v");

    if (cfg.num_v_heads() > 0) {
        alpha_ = backend_.tensor_f32(cfg.num_v_heads(), "recurrent_alpha");
        beta_ = backend_.tensor_f32(cfg.num_v_heads(), "recurrent_beta");
        // Conv1d output (silu(conv(qkv_mixed))) is the same size as the
        // qkv_mixed projection. Allocate once and reuse across all 48
        // recurrent layers and across all tokens.
        if (max_recurrent_qkv > 0) {
            conv_out_ = backend_.tensor_f32(max_recurrent_qkv, "recurrent_conv_out");
        }
    }
    mid_ = backend_.tensor_f32(static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim, "attn_mid");

    // Per-layer DeltaNet state + conv1d ring buffer (only recurrent layers).
    recurrent_states_.resize(weights_.n_layers());
    conv_states_.resize(weights_.n_layers());
    if (cfg.num_v_heads() > 0 && cfg.ssm_conv_kernel > 0) {
        const uint64_t state_size = static_cast<uint64_t>(cfg.num_v_heads())
                                  * cfg.head_v_dim_ssm() * cfg.head_k_dim();
        // conv_dim = key_dim*2 + value_dim. This matches the shape of `proj_`
        // (recurrent_qkv_dim) tracked when we built QwenWeights.
        for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
            const QwenLayerWeights &l = weights_.layer(il);
            if (!l.recurrent) continue;
            const std::string slbl = "recurrent_state_l" + std::to_string(il);
            const std::string clbl = "conv_state_l" + std::to_string(il);
            recurrent_states_[il] = backend_.tensor_f32(state_size, slbl.c_str());
            const uint64_t conv_dim = l.recurrent_qkv_dim; // K+K+V channels
            conv_states_[il] = backend_.tensor_f32(
                conv_dim * (cfg.ssm_conv_kernel - 1), clbl.c_str());
        }
    }
    scores_ = backend_.tensor_f32(static_cast<uint64_t>(cfg.n_heads) * std::max<uint32_t>(kv_ctx_size_, 1), "attn_scores");

    // Per-layer KV cache for the standard-attention layers only.
    const uint64_t kv_per_pos = static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
    const uint64_t kv_physical_slots =
        external_kv_cache_ ? external_kv_cache_->physical_slots
                           : kv_pages_.physical_slots();
    // Default KV cache dtype: FP16 (2x bandwidth at long context, ~equal
    // greedy-token output). Force back to FP32 with QW3_KV_DTYPE=fp32, or down
    // to per-row int8 (one fp16 scale per head_dim row) with QW3_KV_DTYPE=q8.
    const char *kv_dtype_env = std::getenv("QW3_KV_DTYPE");
    const bool kv_use_fp32 = kv_dtype_env && std::strcmp(kv_dtype_env, "fp32") == 0;
    const bool kv_use_q8 = kv_dtype_env && std::strcmp(kv_dtype_env, "q8") == 0;
    const bool kv_use_fp8 = kv_dtype_env && std::strcmp(kv_dtype_env, "fp8") == 0;
    const bool kv_use_fp16 = !kv_use_fp32 && !kv_use_q8 && !kv_use_fp8;
    if (!external_kv_cache_) {
        k_cache_.resize(weights_.n_layers());
        v_cache_.resize(weights_.n_layers());
        for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
            if (!cfg.is_standard_attention_layer(il)) continue;
            const std::string klabel = "k_cache_l" + std::to_string(il);
            const std::string vlabel = "v_cache_l" + std::to_string(il);
            if (kv_use_q8) {
                k_cache_[il] = backend_.tensor_q8_kv(kv_per_pos * kv_physical_slots, cfg.head_dim, klabel.c_str());
                v_cache_[il] = backend_.tensor_q8_kv(kv_per_pos * kv_physical_slots, cfg.head_dim, vlabel.c_str());
            } else if (kv_use_fp8) {
                k_cache_[il] = backend_.tensor_fp8_kv(kv_per_pos * kv_physical_slots, klabel.c_str());
                v_cache_[il] = backend_.tensor_fp8_kv(kv_per_pos * kv_physical_slots, vlabel.c_str());
            } else if (kv_use_fp16) {
                k_cache_[il] = backend_.tensor_f16(kv_per_pos * kv_physical_slots, klabel.c_str());
                v_cache_[il] = backend_.tensor_f16(kv_per_pos * kv_physical_slots, vlabel.c_str());
            } else {
                k_cache_[il] = backend_.tensor_f32(kv_per_pos * kv_physical_slots, klabel.c_str());
                v_cache_[il] = backend_.tensor_f32(kv_per_pos * kv_physical_slots, vlabel.c_str());
            }
        }
    }

    const GgufTensorInfo *head = model_.output();
    logits_ = backend_.tensor_f32(tensor_rows(*head), "logits");

    scratch_ready_ = true;
}

void QwenExecutor::allocate_kvmem_gpu_cache(uint64_t physical_slots) {
    if (scratch_ready_) {
        throw std::runtime_error(
            "KVMem bounded GPU cache must be configured before scratch allocation");
    }
    if (physical_slots == 0) {
        throw std::runtime_error("KVMem bounded GPU cache requires physical_slots > 0");
    }
    const QwenConfig &cfg = model_.config();
    const uint64_t kv_per_pos =
        static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
    const char *kv_dtype_env = std::getenv("QW3_KV_DTYPE");
    const bool kv_use_fp32 = kv_dtype_env && std::strcmp(kv_dtype_env, "fp32") == 0;
    const bool kv_use_q8 = kv_dtype_env && std::strcmp(kv_dtype_env, "q8") == 0;
    const bool kv_use_fp8 = kv_dtype_env && std::strcmp(kv_dtype_env, "fp8") == 0;
    const bool kv_use_fp16 = !kv_use_fp32 && !kv_use_q8 && !kv_use_fp8;

    kvmem_k_cache_storage_.clear();
    kvmem_v_cache_storage_.clear();
    kvmem_k_cache_storage_.resize(weights_.n_layers());
    kvmem_v_cache_storage_.resize(weights_.n_layers());
    kvmem_kv_cache_view_.physical_slots = physical_slots;
    kvmem_kv_cache_view_.k_cache.assign(weights_.n_layers(), nullptr);
    kvmem_kv_cache_view_.v_cache.assign(weights_.n_layers(), nullptr);

    for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
        if (!cfg.is_standard_attention_layer(il)) continue;
        const std::string klabel =
            "kvmem_bounded_k_cache_l" + std::to_string(il);
        const std::string vlabel =
            "kvmem_bounded_v_cache_l" + std::to_string(il);
        if (kv_use_q8) {
            kvmem_k_cache_storage_[il] = backend_.tensor_q8_kv(
                kv_per_pos * physical_slots, cfg.head_dim, klabel.c_str());
            kvmem_v_cache_storage_[il] = backend_.tensor_q8_kv(
                kv_per_pos * physical_slots, cfg.head_dim, vlabel.c_str());
        } else if (kv_use_fp8) {
            kvmem_k_cache_storage_[il] = backend_.tensor_fp8_kv(
                kv_per_pos * physical_slots, klabel.c_str());
            kvmem_v_cache_storage_[il] = backend_.tensor_fp8_kv(
                kv_per_pos * physical_slots, vlabel.c_str());
        } else if (kv_use_fp16) {
            kvmem_k_cache_storage_[il] = backend_.tensor_f16(
                kv_per_pos * physical_slots, klabel.c_str());
            kvmem_v_cache_storage_[il] = backend_.tensor_f16(
                kv_per_pos * physical_slots, vlabel.c_str());
        } else {
            kvmem_k_cache_storage_[il] = backend_.tensor_f32(
                kv_per_pos * physical_slots, klabel.c_str());
            kvmem_v_cache_storage_[il] = backend_.tensor_f32(
                kv_per_pos * physical_slots, vlabel.c_str());
        }
        kvmem_kv_cache_view_.k_cache[il] = kvmem_k_cache_storage_[il].get();
        kvmem_kv_cache_view_.v_cache[il] = kvmem_v_cache_storage_[il].get();
    }
    external_kv_cache_ = &kvmem_kv_cache_view_;
}

std::unique_ptr<DeviceTensor> QwenExecutor::kvmem_alloc_raw_k_tensor(
        uint64_t count, const char *label) {
    const char *dtype = std::getenv("QW3_KV_DTYPE");
    if (dtype && std::strcmp(dtype, "fp32") == 0) {
        return backend_.tensor_f32(count, label);
    }
    if (dtype && std::strcmp(dtype, "fp8") == 0) {
        return backend_.tensor_fp8_kv(count, label);
    }
    if (dtype && std::strcmp(dtype, "q8") == 0) {
        throw std::runtime_error(
            "immutable raw-K capture does not support q8 row-scale storage");
    }
    return backend_.tensor_f16(count, label);
}

std::unique_ptr<DeviceTensor> QwenExecutor::kvmem_alloc_mean_index_tensor(
        uint64_t count, const char *label) {
    const char *dtype = std::getenv("QW3_KV_DTYPE");
    if (dtype && std::strcmp(dtype, "fp32") == 0) {
        return backend_.tensor_f32(count, label);
    }
    // Keep the retrieval index in IEEE binary16 (`__half`, not bfloat16)
    // independently of the active KV-cache dtype. In particular, FP8 K/V may
    // reduce the resident cache footprint, but quantizing the already-averaged
    // mean-K vectors a second time changes retrieval rankings. q8 also has a
    // per-row scale and cannot represent a standalone mean vector through the
    // current tensor interface, so it uses the same FP16 index.
    return backend_.tensor_f16(count, label);
}

bool QwenExecutor::kvmem_cpu_budget_has(uint64_t bytes) const {
    if (!kvmem_sparse_cpu_tier_) return true;
    const uint64_t used = kvmem_raw_k_mirror_bytes_ +
                          kvmem_raw_mtp_k_mirror_bytes_ +
                          kvmem_cpu_sparse_bytes_;
    return used <= kvmem_cpu_budget_bytes_ &&
           bytes <= kvmem_cpu_budget_bytes_ - used;
}

uint8_t *QwenExecutor::kvmem_cpu_slot_data(int32_t slot) {
    if (slot < 0 || !kvmem_cpu_tier_) return nullptr;
    if (kvmem_sparse_cpu_tier_) {
        const size_t i = static_cast<size_t>(slot);
        if (i >= kvmem_cpu_sparse_slot_live_.size() ||
            !kvmem_cpu_sparse_slot_live_[i] ||
            kvmem_cpu_sparse_slot_bytes_ == 0) {
            return nullptr;
        }
        const size_t slab_index =
            i / kvmem_cpu_sparse_slots_per_slab_;
        const size_t slot_in_slab =
            i % kvmem_cpu_sparse_slots_per_slab_;
        if (slab_index >= kvmem_cpu_sparse_slabs_.size() ||
            !kvmem_cpu_sparse_slabs_[slab_index].data) {
            return nullptr;
        }
        return kvmem_cpu_sparse_slabs_[slab_index].data.get() +
               slot_in_slab * kvmem_cpu_sparse_slot_bytes_;
    }
    uint8_t *base = kvmem_cpu_data();
    return base ? base + kvmem_cpu_tier_->slot_offset(slot) : nullptr;
}

const uint8_t *QwenExecutor::kvmem_cpu_slot_data(int32_t slot) const {
    if (slot < 0 || !kvmem_cpu_tier_) return nullptr;
    if (kvmem_sparse_cpu_tier_) {
        const size_t i = static_cast<size_t>(slot);
        if (i >= kvmem_cpu_sparse_slot_live_.size() ||
            !kvmem_cpu_sparse_slot_live_[i] ||
            kvmem_cpu_sparse_slot_bytes_ == 0) {
            return nullptr;
        }
        const size_t slab_index =
            i / kvmem_cpu_sparse_slots_per_slab_;
        const size_t slot_in_slab =
            i % kvmem_cpu_sparse_slots_per_slab_;
        if (slab_index >= kvmem_cpu_sparse_slabs_.size() ||
            !kvmem_cpu_sparse_slabs_[slab_index].data) {
            return nullptr;
        }
        return kvmem_cpu_sparse_slabs_[slab_index].data.get() +
               slot_in_slab * kvmem_cpu_sparse_slot_bytes_;
    }
    const uint8_t *base = kvmem_cpu_data();
    return base ? base + kvmem_cpu_tier_->slot_offset(slot) : nullptr;
}

bool QwenExecutor::kvmem_reserve_cpu_slot(int32_t slot) {
    if (!kvmem_sparse_cpu_tier_) return kvmem_cpu_slot_data(slot) != nullptr;
    if (slot < 0 || !block_store_) return false;
    const size_t i = static_cast<size_t>(slot);
    if (i >= kvmem_cpu_sparse_slot_live_.size() ||
        kvmem_cpu_sparse_slot_bytes_ == 0) {
        return false;
    }
    if (kvmem_cpu_sparse_slot_live_[i]) return true;
    const size_t slab_index =
        i / kvmem_cpu_sparse_slots_per_slab_;
    if (slab_index >= kvmem_cpu_sparse_slabs_.size()) return false;
    KvMemCpuSparseSlab &slab =
        kvmem_cpu_sparse_slabs_[slab_index];
    if (!slab.data) {
        const uint64_t slab_bytes =
            static_cast<uint64_t>(slab.capacity_slots) *
            kvmem_cpu_sparse_slot_bytes_;
        if (slab_bytes == 0 ||
            !kvmem_cpu_budget_has(slab_bytes)) {
            return false;
        }
        // Spill records overwrite every admitted byte. Avoid value-
        // initializing a whole sparse slab on the request thread; first-touch
        // page allocation then happens in the bounded background CPU-copy
        // worker (or in the caller's actual copy for compatibility paths).
        slab.data = std::unique_ptr<uint8_t[]>(
            new uint8_t[static_cast<size_t>(slab_bytes)]);
        kvmem_cpu_sparse_bytes_ += slab_bytes;
    }
    kvmem_cpu_sparse_slot_live_[i] = 1;
    ++slab.live_slots;
    return true;
}

void QwenExecutor::kvmem_release_cpu_slot(int32_t slot) {
    if (!kvmem_sparse_cpu_tier_ || slot < 0 || !block_store_) return;
    const size_t i = static_cast<size_t>(slot);
    if (i >= kvmem_cpu_sparse_slot_live_.size() ||
        !kvmem_cpu_sparse_slot_live_[i]) {
        return;
    }
    const size_t slab_index =
        i / kvmem_cpu_sparse_slots_per_slab_;
    if (slab_index >= kvmem_cpu_sparse_slabs_.size()) return;
    KvMemCpuSparseSlab &slab =
        kvmem_cpu_sparse_slabs_[slab_index];
    kvmem_cpu_sparse_slot_live_[i] = 0;
    if (slab.live_slots > 0) --slab.live_slots;
    if (slab.live_slots == 0 && slab.data) {
        const uint64_t slab_bytes =
            static_cast<uint64_t>(slab.capacity_slots) *
            kvmem_cpu_sparse_slot_bytes_;
        slab.data.reset();
        kvmem_cpu_sparse_bytes_ =
            slab_bytes <= kvmem_cpu_sparse_bytes_
                ? kvmem_cpu_sparse_bytes_ - slab_bytes : 0;
    }
}

void QwenExecutor::kvmem_evict_cpu_for_raw(uint64_t bytes) {
    if (!kvmem_sparse_cpu_tier_ || kvmem_cpu_budget_has(bytes)) return;
    if (!kvmem_cpu_tier_ || !block_store_) {
        throw std::runtime_error(
            "KVMem immutable raw-K growth exhausted --kvmem-cpu-gb; "
            "configure an NVMe tier or raise the CPU budget");
    }
    // A proactive writer may own the destination of a not-yet-published CPU
    // cache copy. Complete those bounded workers before an LRU slot can be
    // recycled by raw-K growth; otherwise the worker could write into freed or
    // reassigned sparse storage.
    kvmem_finish_proactive_d2h(/*wait_all=*/true);
    kvmem_reap_pending_writes(/*wait_all=*/true);
    while (!kvmem_cpu_budget_has(bytes)) {
        // Opt3 can retain empty spill slabs across cold requests. They are
        // allocation-cache capacity, not live KV authority, so raw-K growth may
        // reclaim them before evicting a current request's resident block.
        bool reclaimed_empty_slab = false;
        for (auto it = kvmem_cpu_sparse_slabs_.rbegin();
             it != kvmem_cpu_sparse_slabs_.rend(); ++it) {
            if (it->live_slots != 0 || !it->data) continue;
            const uint64_t slab_bytes =
                static_cast<uint64_t>(it->capacity_slots) *
                kvmem_cpu_sparse_slot_bytes_;
            it->data.reset();
            kvmem_cpu_sparse_bytes_ =
                slab_bytes <= kvmem_cpu_sparse_bytes_
                    ? kvmem_cpu_sparse_bytes_ - slab_bytes
                    : 0;
            reclaimed_empty_slab = true;
            break;
        }
        if (reclaimed_empty_slab) continue;

        // With capacity-qualified CPU-only inclusive backing, a CPU record for
        // a GPU-resident block is merely a clean cache copy. Raw-K authority has
        // priority: discard such a redundant copy without any I/O. This is a
        // fragmentation safety valve (normally the worst-case capacity check
        // means it is never needed).
        if (!kvmem_nvme_tier_ && kvmem_inclusive_cpu_backing_) {
            uint32_t redundant = UINT32_MAX;
            int32_t redundant_slot = -1;
            const auto &blocks = block_store_->blocks();
            for (const KvMemBlock &candidate : blocks) {
                const int32_t slot =
                    kvmem_cpu_tier_->block_slot(candidate.block_id);
                if (slot >= 0 &&
                    kvmem_block_pages_resident(candidate)) {
                    redundant = candidate.block_id;
                    redundant_slot = slot;
                    break;
                }
            }
            if (redundant != UINT32_MAX) {
                kvmem_cpu_tier_->release_block(redundant);
                kvmem_release_cpu_slot(redundant_slot);
                block_store_->set_block_cpu_copy(redundant, -1);
                continue;
            }
        }
        if (!kvmem_nvme_tier_) {
            throw std::runtime_error(
                "KVMem immutable raw-K growth exhausted --kvmem-cpu-gb; "
                "no redundant inclusive CPU copy can be released. Raise the "
                "CPU budget or configure an NVMe tier");
        }
        const int32_t victim_i = kvmem_cpu_tier_->lru_victim();
        if (victim_i < 0) {
            throw std::runtime_error(
                "KVMem immutable raw-K alone exhausted --kvmem-cpu-gb");
        }
        const uint32_t victim = static_cast<uint32_t>(victim_i);
        const int32_t slot = kvmem_cpu_tier_->block_slot(victim);
        const uint8_t *src = kvmem_cpu_slot_data(slot);
        if (slot < 0 || !src ||
            victim >= block_store_->blocks().size()) {
            throw std::runtime_error(
                "KVMem sparse CPU tier lost an LRU block while growing raw-K");
        }
        const uint64_t victim_bytes =
            kvmem_block_spill_bytes(block_store_->blocks()[victim]);
        const bool inclusive =
            block_store_->config().optimization_level >=
            KvMemOptimizationLevel::Opt2;
        const KvMemBlock &victim_block = block_store_->blocks()[victim];
        const bool clean_backing =
            inclusive &&
            (victim_block.ssd_clean || victim_block.in_flight) &&
            victim_block.nvme_slot >= 0 &&
            kvmem_nvme_tier_->block_slot(victim) ==
                victim_block.nvme_slot;
        if (!clean_backing) {
            kvmem_nvme_tier_->write_block(victim, src, victim_bytes);
            if (inclusive) {
                block_store_->set_block_ssd_backing(
                    victim, kvmem_nvme_tier_->block_slot(victim), true);
            }
        }
        kvmem_cpu_tier_->release_block(victim);
        kvmem_release_cpu_slot(slot);
        if (inclusive) {
            block_store_->set_block_cpu_copy(victim, -1);
            if (!kvmem_block_pages_resident(victim_block)) {
                block_store_->set_block_tier(
                    victim, KvTier::SSD, -1,
                    kvmem_nvme_tier_->block_slot(victim));
            }
        } else {
            block_store_->set_block_tier(
                victim, KvTier::SSD, -1,
                kvmem_nvme_tier_->block_slot(victim));
        }
        if (kvmem_tier_trace_enabled()) {
            std::fprintf(
                stderr,
                "[kvmem-tier] raw_k_budget_evict block=%u to=nvme "
                "ssd_write=%d "
                "raw_bytes=%llu cpu_spill_bytes=%llu budget_bytes=%llu\n",
                victim, clean_backing ? 0 : 1,
                static_cast<unsigned long long>(
                    kvmem_raw_k_mirror_bytes_ +
                    kvmem_raw_mtp_k_mirror_bytes_),
                static_cast<unsigned long long>(kvmem_cpu_sparse_bytes_),
                static_cast<unsigned long long>(kvmem_cpu_budget_bytes_));
        }
    }
}

void QwenExecutor::kvmem_ensure_raw_k_chunks(
        uint32_t base, uint32_t rows, bool mtp) {
    if (rows == 0) return;
    const uint64_t end = static_cast<uint64_t>(base) + rows;
    if (end > kv_ctx_size_ || kvmem_raw_k_chunk_tokens_ == 0 ||
        kvmem_raw_k_row_bytes_ == 0) {
        throw std::runtime_error("immutable raw-K chunk range is invalid");
    }
    auto &chunks = mtp ? kvmem_raw_mtp_k_chunks_ : kvmem_raw_k_chunks_;
    const uint32_t first = base / kvmem_raw_k_chunk_tokens_;
    const uint32_t last =
        static_cast<uint32_t>((end - 1) / kvmem_raw_k_chunk_tokens_);
    const uint64_t layers = mtp ? 1 : kvmem_raw_layers_.size();
    const uint64_t chunk_bytes =
        layers * kvmem_raw_k_chunk_tokens_ * kvmem_raw_k_row_bytes_;
    if (last >= chunks.size() ||
        chunk_bytes > static_cast<uint64_t>(
            std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("immutable raw-K chunk allocation is too large");
    }
    for (uint32_t chunk = first; chunk <= last; ++chunk) {
        if (chunks[chunk]) continue;
        kvmem_evict_cpu_for_raw(chunk_bytes);
        if (!kvmem_cpu_budget_has(chunk_bytes)) {
            throw std::runtime_error(
                "immutable raw-K chunk exceeds the configured CPU budget");
        }
        auto storage =
            std::make_unique<uint8_t[]>(static_cast<size_t>(chunk_bytes));
        chunks[chunk] = std::move(storage);
        if (mtp) {
            kvmem_raw_mtp_k_mirror_bytes_ += chunk_bytes;
        } else {
            kvmem_raw_k_mirror_bytes_ += chunk_bytes;
        }
    }
}

void QwenExecutor::kvmem_write_raw_k(
        uint32_t layer_slot, uint32_t base, const uint8_t *src,
        uint32_t rows, bool mtp) {
    if (rows == 0) return;
    kvmem_ensure_raw_k_chunks(base, rows, mtp);
    auto &chunks = mtp ? kvmem_raw_mtp_k_chunks_ : kvmem_raw_k_chunks_;
    const uint32_t bt = block_store_
        ? std::max<uint32_t>(
              1, block_store_->config().block_tokens)
        : 1;
    const uint64_t layers =
        mtp ? 1 : kvmem_raw_layers_.size();
    uint32_t pos = base;
    uint32_t left = rows;
    while (left > 0) {
        const uint32_t chunk = pos / kvmem_raw_k_chunk_tokens_;
        const uint32_t within = pos % kvmem_raw_k_chunk_tokens_;
        uint32_t take =
            std::min(left, kvmem_raw_k_chunk_tokens_ - within);
        uint64_t row_offset = 0;
        if (kvmem_raw_k_block_major_) {
            const uint32_t block = within / bt;
            const uint32_t token = within % bt;
            take = std::min(take, bt - token);
            row_offset =
                (static_cast<uint64_t>(block) * layers +
                 layer_slot) * bt + token;
        } else {
            const uint64_t layer_base = mtp
                ? 0
                : static_cast<uint64_t>(layer_slot) *
                      kvmem_raw_k_chunk_tokens_;
            row_offset = layer_base + within;
        }
        uint8_t *dst = chunks[chunk].get() +
            row_offset * kvmem_raw_k_row_bytes_;
        const uint64_t bytes =
            static_cast<uint64_t>(take) * kvmem_raw_k_row_bytes_;
        std::memcpy(dst, src, static_cast<size_t>(bytes));
        src += bytes;
        pos += take;
        left -= take;
    }
}

void QwenExecutor::kvmem_read_raw_k(
        uint32_t layer_slot, uint32_t base, uint8_t *dst,
        uint32_t rows, bool mtp) const {
    if (rows == 0) return;
    const auto &chunks =
        mtp ? kvmem_raw_mtp_k_chunks_ : kvmem_raw_k_chunks_;
    const uint32_t bt = block_store_
        ? std::max<uint32_t>(
              1, block_store_->config().block_tokens)
        : 1;
    const uint64_t layers =
        mtp ? 1 : kvmem_raw_layers_.size();
    uint32_t pos = base;
    uint32_t left = rows;
    while (left > 0) {
        const uint32_t chunk = pos / kvmem_raw_k_chunk_tokens_;
        const uint32_t within = pos % kvmem_raw_k_chunk_tokens_;
        uint32_t take =
            std::min(left, kvmem_raw_k_chunk_tokens_ - within);
        if (chunk >= chunks.size() || !chunks[chunk]) {
            throw std::runtime_error(
                "immutable raw-K read references an unallocated chunk");
        }
        uint64_t row_offset = 0;
        if (kvmem_raw_k_block_major_) {
            const uint32_t block = within / bt;
            const uint32_t token = within % bt;
            take = std::min(take, bt - token);
            row_offset =
                (static_cast<uint64_t>(block) * layers +
                 layer_slot) * bt + token;
        } else {
            const uint64_t layer_base = mtp
                ? 0
                : static_cast<uint64_t>(layer_slot) *
                      kvmem_raw_k_chunk_tokens_;
            row_offset = layer_base + within;
        }
        const uint8_t *src = chunks[chunk].get() +
            row_offset * kvmem_raw_k_row_bytes_;
        const uint64_t bytes =
            static_cast<uint64_t>(take) * kvmem_raw_k_row_bytes_;
        std::memcpy(dst, src, static_cast<size_t>(bytes));
        dst += bytes;
        pos += take;
        left -= take;
    }
}

void QwenExecutor::kvmem_truncate_raw_k(uint32_t token_pos) {
    auto release_suffix = [&](auto &chunks, uint64_t chunk_bytes,
                              uint64_t &allocated_bytes) {
        const uint32_t first =
            (token_pos + kvmem_raw_k_chunk_tokens_ - 1) /
            kvmem_raw_k_chunk_tokens_;
        for (uint32_t i = first; i < chunks.size(); ++i) {
            if (!chunks[i]) continue;
            chunks[i].reset();
            allocated_bytes = chunk_bytes <= allocated_bytes
                ? allocated_bytes - chunk_bytes
                : 0;
        }
    };
    if (kvmem_raw_k_chunk_tokens_ == 0) return;
    const uint64_t main_chunk_bytes =
        static_cast<uint64_t>(kvmem_raw_layers_.size()) *
        kvmem_raw_k_chunk_tokens_ * kvmem_raw_k_row_bytes_;
    const uint64_t mtp_chunk_bytes =
        static_cast<uint64_t>(kvmem_raw_k_chunk_tokens_) *
        kvmem_raw_k_row_bytes_;
    release_suffix(kvmem_raw_k_chunks_, main_chunk_bytes,
                   kvmem_raw_k_mirror_bytes_);
    release_suffix(kvmem_raw_mtp_k_chunks_, mtp_chunk_bytes,
                   kvmem_raw_mtp_k_mirror_bytes_);
}

void QwenExecutor::kvmem_ensure_raw_capture_capacity(uint32_t rows) {
    if (!kvmem_immutable_source_k_ || rows == 0) return;
    if (rows <= kvmem_raw_capture_rows_ &&
        kvmem_raw_capture_dev_.size() == kvmem_raw_layers_.size() &&
        kvmem_raw_capture_host_) {
        return;
    }
    const QwenConfig &cfg = model_.config();
    const uint64_t per_pos =
        static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
    kvmem_raw_capture_rows_ = std::max(rows, kvmem_raw_capture_rows_);
    kvmem_raw_capture_dev_.clear();
    kvmem_raw_capture_dev_.reserve(kvmem_raw_layers_.size());
    for (uint32_t slot = 0; slot < kvmem_raw_layers_.size(); ++slot) {
        const std::string label =
            "kvmem_raw_k_capture_l" +
            std::to_string(kvmem_raw_layers_[slot]);
        kvmem_raw_capture_dev_.push_back(kvmem_alloc_raw_k_tensor(
            static_cast<uint64_t>(kvmem_raw_capture_rows_) * per_pos,
            label.c_str()));
    }
    const uint64_t host_bytes =
        static_cast<uint64_t>(kvmem_raw_layers_.size()) *
        kvmem_raw_capture_rows_ * kvmem_raw_k_row_bytes_;
    kvmem_raw_capture_host_ =
        backend_.host_buffer(host_bytes, "kvmem_raw_k_capture_host");
}

void QwenExecutor::kvmem_capture_raw_k_batch(
        uint32_t layer, const DeviceTensor &raw_k, uint32_t batch) {
    if (!kvmem_immutable_source_k_ || batch == 0 ||
        layer >= kvmem_raw_layer_slot_.size()) {
        return;
    }
    const int32_t slot = kvmem_raw_layer_slot_[layer];
    if (slot < 0) return;
    const QwenConfig &cfg = model_.config();
    const uint32_t per_pos = cfg.n_kv_heads * cfg.head_dim;
    require_status(backend_.kv_append_batch(
        *kvmem_raw_capture_dev_[static_cast<uint32_t>(slot)], raw_k,
        /*base_pos=*/0, per_pos, batch));
}

void QwenExecutor::kvmem_capture_raw_k_decode(
        uint32_t layer, const DeviceTensor &raw_k, uint32_t row) {
    if (!kvmem_immutable_source_k_ ||
        layer >= kvmem_raw_layer_slot_.size()) {
        return;
    }
    const int32_t slot = kvmem_raw_layer_slot_[layer];
    if (slot < 0) return;
    const QwenConfig &cfg = model_.config();
    const uint32_t per_pos = cfg.n_kv_heads * cfg.head_dim;
    require_status(backend_.kv_append(
        *kvmem_raw_capture_dev_[static_cast<uint32_t>(slot)], raw_k,
        row, per_pos));
}

void QwenExecutor::kvmem_flush_raw_k_capture(
        uint32_t true_base, uint32_t first_row, uint32_t rows) {
    if (!kvmem_immutable_source_k_ || rows == 0) return;
    if (!kvmem_raw_capture_host_ ||
        true_base + rows > kv_ctx_size_ ||
        first_row + rows > kvmem_raw_capture_rows_) {
        throw std::runtime_error("immutable raw-K capture range is invalid");
    }
    const uint64_t row_bytes = kvmem_raw_k_row_bytes_;
    const uint64_t layer_bytes = static_cast<uint64_t>(rows) * row_bytes;
    require_status(backend_.begin_kv_transfer_from_device());
    for (uint32_t slot = 0; slot < kvmem_raw_layers_.size(); ++slot) {
        require_status(backend_.copy_bytes_to_host_async(
            *kvmem_raw_capture_dev_[slot],
            static_cast<uint8_t *>(kvmem_raw_capture_host_->data) +
                static_cast<uint64_t>(slot) * layer_bytes,
            static_cast<uint64_t>(first_row) * row_bytes, layer_bytes));
    }
    require_status(backend_.wait_kv_transfer());
    for (uint32_t slot = 0; slot < kvmem_raw_layers_.size(); ++slot) {
        const uint8_t *src =
            static_cast<const uint8_t *>(kvmem_raw_capture_host_->data) +
            static_cast<uint64_t>(slot) * layer_bytes;
        kvmem_write_raw_k(slot, true_base, src, rows, /*mtp=*/false);
    }
    std::fill(kvmem_raw_k_valid_tokens_.begin() + true_base,
              kvmem_raw_k_valid_tokens_.begin() + true_base + rows,
              static_cast<uint8_t>(1));
}

void QwenExecutor::kvmem_flush_raw_k_decode() {
    if (!kvmem_immutable_source_k_ ||
        kvmem_raw_decode_block_start_ < 0 ||
        kvmem_raw_decode_rows_ == 0) {
        return;
    }
    const uint32_t true_base =
        static_cast<uint32_t>(kvmem_raw_decode_block_start_) +
        kvmem_raw_decode_first_row_;
    kvmem_flush_raw_k_capture(true_base, kvmem_raw_decode_first_row_,
                              kvmem_raw_decode_rows_);
    kvmem_raw_decode_block_start_ = -1;
    kvmem_raw_decode_first_row_ = 0;
    kvmem_raw_decode_rows_ = 0;
}

void QwenExecutor::kvmem_ensure_rope_sincos_table() {
    if (!kvmem_rope_table_enabled_ || kvmem_rope_sincos_) return;
    const QwenConfig &cfg = model_.config();
    if (kvmem_rope_table_positions_ == 0 || cfg.rope_dim == 0 ||
        (cfg.rope_dim % 2u) != 0) {
        throw std::runtime_error("KVMem RoPE table has an invalid shape");
    }
    const uint64_t floats =
        static_cast<uint64_t>(kvmem_rope_table_positions_) *
        (cfg.rope_dim / 2u) * 2u;
    auto table =
        backend_.scratch_f32(floats, "kvmem_rope_sincos_table");
    const DeviceStatus st = backend_.build_rope_sincos_table_device(
        *table, kvmem_rope_table_positions_, cfg.rope_dim, cfg.rope_theta);
    if (!st.ok) {
        if (kvmem_rope_table_explicit_) {
            throw std::runtime_error(
                std::string("explicit KVMem table assembly failed: ") +
                st.message);
        }
        kvmem_rope_table_enabled_ = false;
        kvmem_rope_table_positions_ = 0;
        std::fprintf(
            stderr,
            "[kvmem-assembly] mode=legacy table_fallback_reason=%s\n",
            st.message);
        return;
    }
    kvmem_rope_sincos_ = std::move(table);
    std::fprintf(
        stderr,
        "[kvmem-assembly] mode=%s positions=%u rope_pairs=%u "
        "table_bytes=%llu raw_transfer_buffers=%u\n",
        kvmem_raw_pipeline_enabled_ ? "pipeline" : "table",
        kvmem_rope_table_positions_, cfg.rope_dim / 2u,
        static_cast<unsigned long long>(floats * sizeof(float)),
        kvmem_raw_pipeline_enabled_ ? 2u : 1u);
}

void QwenExecutor::kvmem_materialize_raw_k(
        const std::vector<const KvMemRemap *> &refreshes) {
    if (refreshes.empty()) return;
    if (kvmem_raw_k_chunks_.empty() || !window_pages_device_) {
        throw std::runtime_error("immutable raw-K materialization is unavailable");
    }
    const QwenConfig &cfg = model_.config();
    const uint32_t bt = block_store_->config().block_tokens;
    const uint64_t per_pos =
        static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
    const uint64_t block_elems = static_cast<uint64_t>(bt) * per_pos;
    const uint64_t block_bytes = static_cast<uint64_t>(bt) *
                                 kvmem_raw_k_row_bytes_;
    const uint32_t cap = std::max<uint32_t>(
        1, kvmem_raw_transfer_block_cap_);

    for (uint32_t begin = 0; begin < refreshes.size(); begin += cap) {
        RawMaterializeSlot *pipeline_slot = nullptr;
        if (kvmem_raw_pipeline_enabled_) {
            pipeline_slot =
                &kvmem_raw_pipeline_slots_[(begin / cap) % 2u];
            const uint64_t wait0 = kvmem_steady_ns();
            require_status(backend_.wait_kv_transfer_fence(
                pipeline_slot->h2d_done));
            kvmem_assembly_raw_h2d_wait_ns_ +=
                kvmem_steady_ns() - wait0;
            require_status(backend_.kv_transfer_wait_for_execution(
                pipeline_slot->compute_done));
        }
        const uint32_t n = std::min<uint32_t>(
            cap, static_cast<uint32_t>(refreshes.size()) - begin);
        const uint64_t total_elems =
            static_cast<uint64_t>(kvmem_raw_layers_.size()) * n * block_elems;
        const uint64_t total_bytes =
            static_cast<uint64_t>(kvmem_raw_layers_.size()) * n * block_bytes;
        DeviceTensor *transfer_dev = nullptr;
        HostBuffer *transfer_host = nullptr;
        if (pipeline_slot) {
            const uint64_t capacity_elems =
                static_cast<uint64_t>(kvmem_raw_layers_.size()) *
                cap * block_elems;
            const uint64_t capacity_bytes =
                static_cast<uint64_t>(kvmem_raw_layers_.size()) *
                cap * block_bytes;
            if (!pipeline_slot->device) {
                pipeline_slot->device = kvmem_alloc_raw_k_tensor(
                    capacity_elems, "kvmem_raw_k_pipeline_device");
                pipeline_slot->host = backend_.host_buffer(
                    capacity_bytes, "kvmem_raw_k_pipeline_host");
                pipeline_slot->device_elements = capacity_elems;
                pipeline_slot->host_bytes = capacity_bytes;
                // Typed tensor construction zero-initializes on exec_stream.
                // Order the first copy-stream overwrite after that init.
                require_status(backend_.record_execution_fence(
                    pipeline_slot->compute_done));
                require_status(backend_.kv_transfer_wait_for_execution(
                    pipeline_slot->compute_done));
            }
            transfer_dev = pipeline_slot->device.get();
            transfer_host = pipeline_slot->host.get();
        } else {
            if (!kvmem_raw_transfer_dev_ ||
                kvmem_raw_transfer_dev_->count < total_elems) {
                kvmem_raw_transfer_dev_ =
                    kvmem_alloc_raw_k_tensor(
                        total_elems, "kvmem_raw_k_transfer_device");
                kvmem_raw_transfer_host_ =
                    backend_.host_buffer(
                        total_bytes, "kvmem_raw_k_transfer_host");
                kvmem_raw_transfer_blocks_ = n;
            }
            transfer_dev = kvmem_raw_transfer_dev_.get();
            transfer_host = kvmem_raw_transfer_host_.get();
        }
        uint8_t *packed =
            static_cast<uint8_t *>(transfer_host->data);
        const uint64_t gather0 = kvmem_steady_ns();
        bs_remap_to_host_.clear();
        bs_remap_ntok_host_.clear();
        uint32_t max_tokens = 0;
        // Validity is block-global, not layer-specific. The legacy loop
        // repeated this check for every standard-attention layer.
        for (uint32_t j = 0; j < n; ++j) {
            const KvMemRemap &rm = *refreshes[begin + j];
            const KvMemBlock &block =
                block_store_->blocks()[rm.block_id];
            for (uint32_t t = 0; t < rm.n_tokens; ++t) {
                const uint32_t pos = block.orig_pos_start + t;
                if (pos >= kvmem_raw_k_valid_tokens_.size() ||
                    !kvmem_raw_k_valid_tokens_[pos]) {
                    throw std::runtime_error(
                        "immutable raw-K block contains uncaptured token " +
                        std::to_string(pos));
                }
            }
        }
        auto gather_layer = [&](uint32_t slot) {
            for (uint32_t j = 0; j < n; ++j) {
                const KvMemRemap &rm = *refreshes[begin + j];
                const KvMemBlock &block =
                    block_store_->blocks()[rm.block_id];
                uint8_t *dst = packed +
                    (static_cast<uint64_t>(slot) * n + j) * block_bytes;
                const uint64_t valid_bytes =
                    static_cast<uint64_t>(rm.n_tokens) *
                    kvmem_raw_k_row_bytes_;
                kvmem_read_raw_k(
                    slot, block.orig_pos_start, dst, rm.n_tokens,
                    /*mtp=*/false);
                if (valid_bytes < block_bytes) {
                    std::memset(dst + valid_bytes, 0,
                                static_cast<size_t>(block_bytes - valid_bytes));
                }
            }
        };
        // Optimized CPU authority and transfer slab share a block-major
        // layout. A full selected block is one contiguous ~1 MiB memcpy
        // instead of L independent 64 KiB copies spread across a chunk.
        auto gather_block = [&](uint32_t j) {
            const KvMemRemap &rm = *refreshes[begin + j];
            const KvMemBlock &block =
                block_store_->blocks()[rm.block_id];
            uint8_t *dst = packed +
                static_cast<uint64_t>(j) *
                    kvmem_raw_layers_.size() * block_bytes;
            const uint32_t chunk =
                block.orig_pos_start / kvmem_raw_k_chunk_tokens_;
            const uint32_t within =
                block.orig_pos_start % kvmem_raw_k_chunk_tokens_;
            const bool contiguous_full_block =
                rm.n_tokens == bt && within % bt == 0 &&
                within + bt <= kvmem_raw_k_chunk_tokens_ &&
                chunk < kvmem_raw_k_chunks_.size() &&
                kvmem_raw_k_chunks_[chunk] != nullptr;
            if (contiguous_full_block) {
                const uint64_t block_in_chunk = within / bt;
                const uint8_t *src =
                    kvmem_raw_k_chunks_[chunk].get() +
                    block_in_chunk * kvmem_raw_layers_.size() *
                        block_bytes;
                std::memcpy(
                    dst, src,
                    static_cast<size_t>(
                        kvmem_raw_layers_.size() * block_bytes));
                return;
            }
            for (uint32_t slot = 0;
                 slot < kvmem_raw_layers_.size(); ++slot) {
                uint8_t *layer_dst =
                    dst + static_cast<uint64_t>(slot) * block_bytes;
                const uint64_t valid_bytes =
                    static_cast<uint64_t>(rm.n_tokens) *
                    kvmem_raw_k_row_bytes_;
                kvmem_read_raw_k(
                    slot, block.orig_pos_start, layer_dst, rm.n_tokens,
                    /*mtp=*/false);
                if (valid_bytes < block_bytes) {
                    std::memset(
                        layer_dst + valid_bytes, 0,
                        static_cast<size_t>(block_bytes - valid_bytes));
                }
            }
        };
        const uint32_t work_items = kvmem_raw_k_block_major_
            ? n
            : static_cast<uint32_t>(kvmem_raw_layers_.size());
        const uint32_t gather_workers = kvmem_raw_pipeline_enabled_
            ? std::min<uint32_t>(
                  static_cast<uint32_t>(
                      kvmem_cpu_gather_threads()),
                  work_items)
            : 1u;
        auto gather_item = [&](uint32_t item) {
            if (kvmem_raw_k_block_major_) {
                gather_block(item);
            } else {
                gather_layer(item);
            }
        };
        if (gather_workers <= 1u) {
            for (uint32_t item = 0; item < work_items; ++item) {
                gather_item(item);
            }
        } else if (kvmem_cpu_worker_pool_) {
            kvmem_cpu_worker_pool_->run(
                work_items, gather_workers,
                [&](size_t item) {
                    gather_item(static_cast<uint32_t>(item));
                });
        } else {
            std::atomic<uint32_t> next_item{0};
            std::exception_ptr gather_error;
            std::mutex gather_error_mu;
            auto worker = [&]() {
                try {
                    for (;;) {
                        const uint32_t item =
                            next_item.fetch_add(
                                1, std::memory_order_relaxed);
                        if (item >= work_items) break;
                        gather_item(item);
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lock(gather_error_mu);
                    if (!gather_error) {
                        gather_error = std::current_exception();
                    }
                }
            };
            std::vector<std::thread> threads;
            threads.reserve(gather_workers - 1u);
            for (uint32_t i = 1; i < gather_workers; ++i) {
                threads.emplace_back(worker);
            }
            worker();
            for (auto &thread : threads) thread.join();
            if (gather_error) std::rethrow_exception(gather_error);
        }
        kvmem_assembly_raw_gather_ns_ +=
            kvmem_steady_ns() - gather0;
        for (uint32_t j = 0; j < n; ++j) {
            const KvMemRemap &rm = *refreshes[begin + j];
            bs_remap_to_host_.push_back(rm.to_base);
            bs_remap_ntok_host_.push_back(
                static_cast<int32_t>(rm.n_tokens));
            max_tokens = std::max(max_tokens, rm.n_tokens);
        }
        if (n > bs_remap_capacity_) {
            bs_remap_capacity_ = n;
            bs_remap_to_dev_ =
                backend_.tensor_i32(n, "bs_remap_to");
            bs_remap_from_dev_ =
                backend_.tensor_i32(n, "bs_remap_from");
            bs_remap_ntok_dev_ =
                backend_.tensor_i32(n, "bs_remap_ntok");
        }
        require_status(backend_.copy_i32_from_host(
            *bs_remap_to_dev_, 0, bs_remap_to_host_.data(), n));
        require_status(backend_.copy_i32_from_host(
            *bs_remap_ntok_dev_, 0, bs_remap_ntok_host_.data(), n));
        const uint64_t h2d0 = kvmem_steady_ns();
        if (pipeline_slot) {
            require_status(backend_.begin_kv_transfer_to_device());
            require_status(backend_.copy_bytes_from_host_async(
                *transfer_dev, 0, packed, total_bytes));
            require_status(backend_.record_kv_transfer_fence(
                pipeline_slot->h2d_done));
            require_status(backend_.execution_wait_for_kv_transfer(
                pipeline_slot->h2d_done));
        } else {
            require_status(backend_.copy_bytes_from_host(
                *transfer_dev, 0, packed, total_bytes));
        }
        kvmem_assembly_raw_h2d_submit_ns_ +=
            kvmem_steady_ns() - h2d0;
        kvmem_assembly_raw_bytes_ += total_bytes;
        ++kvmem_assembly_raw_batches_;
        if (std::getenv("QW3_KVMEM_TRACE") ||
            kvmem_tier_trace_enabled()) {
            std::fprintf(
                stderr,
                "[kvmem-raw-k] refresh_blocks=%u packed_bytes=%llu "
                "first_block=%u\n",
                n, static_cast<unsigned long long>(total_bytes),
                refreshes[begin]->block_id);
        }
        for (uint32_t slot = 0; slot < kvmem_raw_layers_.size(); ++slot) {
            const uint64_t raw_element_offset =
                kvmem_raw_k_block_major_
                    ? static_cast<uint64_t>(slot) * block_elems
                    : static_cast<uint64_t>(slot) * n * block_elems;
            const uint64_t raw_block_stride =
                kvmem_raw_k_block_major_
                    ? static_cast<uint64_t>(
                          kvmem_raw_layers_.size()) * block_elems
                    : 0;
            require_status(
                backend_.raw_k_scatter_rope_paged_batched_device(
                    k_cache(kvmem_raw_layers_[slot]),
                    *transfer_dev,
                    raw_element_offset,
                    n, bt, cfg.n_kv_heads,
                    static_cast<uint32_t>(per_pos), cfg.head_dim,
                    cfg.rope_dim, *bs_remap_to_dev_,
                    *bs_remap_ntok_dev_, *window_pages_device_,
                    kv_pages_.page_size, cfg.rope_theta,
                    kvmem_rope_sincos_.get(),
                    kvmem_rope_table_positions_,
                    raw_block_stride));
        }
        if (pipeline_slot) {
            require_status(backend_.record_execution_fence(
                pipeline_slot->compute_done));
        }
    }
}

void QwenExecutor::kvmem_ensure_raw_mtp_capture_capacity(uint32_t rows) {
    if (!kvmem_mtp_local_positions_ || rows == 0) return;
    if (rows <= kvmem_raw_mtp_capture_rows_ &&
        kvmem_raw_mtp_capture_dev_ && kvmem_raw_mtp_capture_host_) {
        return;
    }
    const QwenConfig &cfg = model_.config();
    const uint64_t per_pos =
        static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
    kvmem_raw_mtp_capture_rows_ =
        std::max(rows, kvmem_raw_mtp_capture_rows_);
    kvmem_raw_mtp_capture_dev_ = kvmem_alloc_raw_k_tensor(
        static_cast<uint64_t>(kvmem_raw_mtp_capture_rows_) * per_pos,
        "kvmem_raw_mtp_k_capture");
    kvmem_raw_mtp_capture_host_ = backend_.host_buffer(
        static_cast<uint64_t>(kvmem_raw_mtp_capture_rows_) *
            kvmem_raw_k_row_bytes_,
        "kvmem_raw_mtp_k_capture_host");
}

void QwenExecutor::kvmem_capture_raw_mtp_k(
        const DeviceTensor &raw_k,
        uint32_t logical_base,
        uint32_t rows,
        uint32_t src_row) {
    if (!kvmem_mtp_local_positions_ || rows == 0) return;
    if (src_row != 0) {
        throw std::runtime_error(
            "immutable MTP raw-K capture requires a zero-based source batch");
    }
    if (logical_base + rows > kv_ctx_size_) {
        throw std::runtime_error(
            "immutable MTP raw-K capture range is invalid");
    }
    kvmem_ensure_raw_mtp_capture_capacity(rows);
    const QwenConfig &cfg = model_.config();
    const uint32_t per_pos = cfg.n_kv_heads * cfg.head_dim;
    require_status(backend_.kv_append_batch(
        *kvmem_raw_mtp_capture_dev_, raw_k,
        /*base_pos=*/0, per_pos, rows));
    const uint64_t bytes =
        static_cast<uint64_t>(rows) * kvmem_raw_k_row_bytes_;
    require_status(backend_.begin_kv_transfer_from_device());
    require_status(backend_.copy_bytes_to_host_async(
        *kvmem_raw_mtp_capture_dev_,
        kvmem_raw_mtp_capture_host_->data, 0, bytes));
    require_status(backend_.wait_kv_transfer());
    kvmem_write_raw_k(
        /*layer_slot=*/0, logical_base,
        static_cast<const uint8_t *>(kvmem_raw_mtp_capture_host_->data),
        rows, /*mtp=*/true);
    std::fill(
        kvmem_raw_mtp_k_valid_tokens_.begin() + logical_base,
        kvmem_raw_mtp_k_valid_tokens_.begin() + logical_base + rows,
        static_cast<uint8_t>(1));
}

void QwenExecutor::kvmem_materialize_raw_mtp_k(
        const KvMemPlan &plan) {
    if (!kvmem_mtp_local_positions_ || plan.remaps.empty()) return;
    if (kvmem_raw_mtp_k_chunks_.empty() || !mtp_window_pages_device_) {
        throw std::runtime_error(
            "immutable MTP raw-K materialization is unavailable");
    }
    const QwenConfig &cfg = model_.config();
    const uint32_t bt = block_store_->config().block_tokens;
    const uint64_t per_pos =
        static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
    const uint64_t block_elems = static_cast<uint64_t>(bt) * per_pos;
    const uint64_t block_bytes =
        static_cast<uint64_t>(bt) * kvmem_raw_k_row_bytes_;
    const uint32_t base_cap = std::max<uint32_t>(
        1, kvmem_raw_transfer_block_cap_);
    // Reuse the main pipeline's byte-sized slots. One MTP block carries only
    // one layer, so the same ~128 MiB slot can hold standard_layer_count times
    // as many blocks and avoids 50 tiny synchronized H2D batches.
    const uint32_t cap = kvmem_raw_pipeline_enabled_
        ? base_cap * std::max<uint32_t>(
              1, static_cast<uint32_t>(kvmem_raw_layers_.size()))
        : base_cap;
    const auto &blocks = block_store_->blocks();

    for (uint32_t begin = 0; begin < plan.remaps.size(); begin += cap) {
        RawMaterializeSlot *pipeline_slot = nullptr;
        if (kvmem_raw_pipeline_enabled_) {
            pipeline_slot =
                &kvmem_raw_pipeline_slots_[(begin / cap) % 2u];
            const uint64_t wait0 = kvmem_steady_ns();
            require_status(backend_.wait_kv_transfer_fence(
                pipeline_slot->h2d_done));
            kvmem_assembly_raw_h2d_wait_ns_ +=
                kvmem_steady_ns() - wait0;
            require_status(backend_.kv_transfer_wait_for_execution(
                pipeline_slot->compute_done));
        }
        const uint32_t n = std::min<uint32_t>(
            cap, static_cast<uint32_t>(plan.remaps.size()) - begin);
        const uint64_t elems = static_cast<uint64_t>(n) * block_elems;
        const uint64_t bytes = static_cast<uint64_t>(n) * block_bytes;
        DeviceTensor *transfer_dev = nullptr;
        HostBuffer *transfer_host = nullptr;
        if (pipeline_slot) {
            const uint64_t capacity_elems =
                static_cast<uint64_t>(
                    std::max<uint32_t>(
                        1, static_cast<uint32_t>(
                               kvmem_raw_layers_.size()))) *
                base_cap * block_elems;
            const uint64_t capacity_bytes =
                static_cast<uint64_t>(
                    std::max<uint32_t>(
                        1, static_cast<uint32_t>(
                               kvmem_raw_layers_.size()))) *
                base_cap * block_bytes;
            if (!pipeline_slot->device) {
                pipeline_slot->device = kvmem_alloc_raw_k_tensor(
                    capacity_elems, "kvmem_raw_k_pipeline_device");
                pipeline_slot->host = backend_.host_buffer(
                    capacity_bytes, "kvmem_raw_k_pipeline_host");
                pipeline_slot->device_elements = capacity_elems;
                pipeline_slot->host_bytes = capacity_bytes;
                require_status(backend_.record_execution_fence(
                    pipeline_slot->compute_done));
                require_status(backend_.kv_transfer_wait_for_execution(
                    pipeline_slot->compute_done));
            }
            if (pipeline_slot->device->count < elems ||
                !pipeline_slot->host ||
                pipeline_slot->host->bytes < bytes) {
                throw std::runtime_error(
                    "KVMem raw MTP pipeline slot is smaller than its "
                    "main-layer byte capacity");
            }
            transfer_dev = pipeline_slot->device.get();
            transfer_host = pipeline_slot->host.get();
        } else {
            if (!kvmem_raw_mtp_transfer_dev_ ||
                kvmem_raw_mtp_transfer_dev_->count < elems) {
                kvmem_raw_mtp_transfer_dev_ = kvmem_alloc_raw_k_tensor(
                    elems, "kvmem_raw_mtp_k_transfer_device");
                kvmem_raw_mtp_transfer_host_ = backend_.host_buffer(
                    bytes, "kvmem_raw_mtp_k_transfer_host");
            }
            transfer_dev = kvmem_raw_mtp_transfer_dev_.get();
            transfer_host = kvmem_raw_mtp_transfer_host_.get();
        }

        uint8_t *packed = static_cast<uint8_t *>(
            transfer_host->data);
        const uint64_t gather0 = kvmem_steady_ns();
        bs_mtp_remap_to_host_.clear();
        bs_mtp_remap_ntok_host_.clear();
        uint32_t max_tokens = 0;
        for (uint32_t j = 0; j < n; ++j) {
            const KvMemRemap &rm = plan.remaps[begin + j];
            const KvMemBlock &block = blocks[rm.block_id];
            if (cfg.n_ctx_train > 0 &&
                static_cast<uint64_t>(rm.to_base) + rm.n_tokens >
                    cfg.n_ctx_train) {
                throw std::runtime_error(
                    "immutable MTP raw-K materialization exceeds the model "
                    "context limit");
            }
            for (uint32_t t = 0; t < rm.n_tokens; ++t) {
                const uint32_t pos = block.orig_pos_start + t;
                if (pos >= kvmem_raw_mtp_k_valid_tokens_.size() ||
                    !kvmem_raw_mtp_k_valid_tokens_[pos]) {
                    throw std::runtime_error(
                        "immutable MTP raw-K block contains uncaptured token " +
                        std::to_string(pos));
                }
            }
            bs_mtp_remap_to_host_.push_back(rm.to_base);
            bs_mtp_remap_ntok_host_.push_back(
                static_cast<int32_t>(rm.n_tokens));
            max_tokens = std::max(max_tokens, rm.n_tokens);
        }
        auto gather_block = [&](uint32_t j) {
            const KvMemRemap &rm = plan.remaps[begin + j];
            const KvMemBlock &block = blocks[rm.block_id];
            uint8_t *dst =
                packed + static_cast<uint64_t>(j) * block_bytes;
            const uint64_t valid_bytes =
                static_cast<uint64_t>(rm.n_tokens) *
                kvmem_raw_k_row_bytes_;
            kvmem_read_raw_k(
                /*layer_slot=*/0, block.orig_pos_start, dst,
                rm.n_tokens, /*mtp=*/true);
            if (valid_bytes < block_bytes) {
                std::memset(dst + valid_bytes, 0,
                            static_cast<size_t>(block_bytes - valid_bytes));
            }
        };
        const uint32_t gather_workers = kvmem_raw_pipeline_enabled_
            ? std::min<uint32_t>(
                  static_cast<uint32_t>(
                      kvmem_cpu_gather_threads()),
                  n)
            : 1u;
        if (gather_workers <= 1u) {
            for (uint32_t j = 0; j < n; ++j) gather_block(j);
        } else if (kvmem_cpu_worker_pool_) {
            kvmem_cpu_worker_pool_->run(
                n, gather_workers,
                [&](size_t j) {
                    gather_block(static_cast<uint32_t>(j));
                });
        } else {
            std::atomic<uint32_t> next_block{0};
            std::exception_ptr gather_error;
            std::mutex gather_error_mu;
            auto worker = [&]() {
                try {
                    for (;;) {
                        const uint32_t j = next_block.fetch_add(
                            1, std::memory_order_relaxed);
                        if (j >= n) break;
                        gather_block(j);
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lock(gather_error_mu);
                    if (!gather_error) {
                        gather_error = std::current_exception();
                    }
                }
            };
            std::vector<std::thread> threads;
            threads.reserve(gather_workers - 1u);
            for (uint32_t i = 1; i < gather_workers; ++i) {
                threads.emplace_back(worker);
            }
            worker();
            for (auto &thread : threads) thread.join();
            if (gather_error) std::rethrow_exception(gather_error);
        }
        kvmem_assembly_raw_gather_ns_ +=
            kvmem_steady_ns() - gather0;
        if (n > bs_mtp_remap_capacity_) {
            bs_mtp_remap_capacity_ = n;
            bs_mtp_remap_to_dev_ = backend_.tensor_i32(
                n, "bs_mtp_remap_to");
            bs_mtp_remap_from_dev_ = backend_.tensor_i32(
                n, "bs_mtp_remap_from");
            bs_mtp_remap_ntok_dev_ = backend_.tensor_i32(
                n, "bs_mtp_remap_ntok");
        }
        require_status(backend_.copy_i32_from_host(
            *bs_mtp_remap_to_dev_, 0,
            bs_mtp_remap_to_host_.data(), n));
        require_status(backend_.copy_i32_from_host(
            *bs_mtp_remap_ntok_dev_, 0,
            bs_mtp_remap_ntok_host_.data(), n));
        const uint64_t h2d0 = kvmem_steady_ns();
        if (pipeline_slot) {
            require_status(backend_.begin_kv_transfer_to_device());
            require_status(backend_.copy_bytes_from_host_async(
                *transfer_dev, 0, packed, bytes));
            require_status(backend_.record_kv_transfer_fence(
                pipeline_slot->h2d_done));
            require_status(backend_.execution_wait_for_kv_transfer(
                pipeline_slot->h2d_done));
        } else {
            require_status(backend_.copy_bytes_from_host(
                *transfer_dev, 0, packed, bytes));
        }
        kvmem_assembly_raw_h2d_submit_ns_ +=
            kvmem_steady_ns() - h2d0;
        kvmem_assembly_raw_bytes_ += bytes;
        ++kvmem_assembly_raw_batches_;
        require_status(
            backend_.raw_k_scatter_rope_paged_batched_device(
                mtp_k_cache(), *transfer_dev,
                /*src_elem_offset=*/0, n, bt, cfg.n_kv_heads,
                static_cast<uint32_t>(per_pos), cfg.head_dim,
                cfg.rope_dim, *bs_mtp_remap_to_dev_,
                *bs_mtp_remap_ntok_dev_,
                *mtp_window_pages_device_,
                mtp_kv_pages_.page_size, cfg.rope_theta,
                kvmem_rope_sincos_.get(),
                kvmem_rope_table_positions_));
        if (pipeline_slot) {
            require_status(backend_.record_execution_fence(
                pipeline_slot->compute_done));
        }
        if (std::getenv("QW3_KVMEM_TRACE") ||
            kvmem_tier_trace_enabled()) {
            std::fprintf(
                stderr,
                "[kvmem-raw-mtp-k] refresh_blocks=%u packed_bytes=%llu "
                "first_block=%u max_rope_end=%u\n",
                n, static_cast<unsigned long long>(bytes),
                plan.remaps[begin].block_id,
                bs_mtp_remap_to_host_.back() + max_tokens);
        }
    }
}

void QwenExecutor::allocate_kvmem_mtp_gpu_cache(uint64_t physical_slots) {
    if (mtp_scratch_ready_) {
        throw std::runtime_error(
            "KVMem bounded MTP GPU cache must be configured before MTP scratch allocation");
    }
    if (physical_slots == 0) {
        throw std::runtime_error("KVMem bounded MTP GPU cache requires physical_slots > 0");
    }
    const QwenConfig &cfg = model_.config();
    const uint64_t kv_per_pos =
        static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
    const char *kv_dtype_env = std::getenv("QW3_KV_DTYPE");
    const bool kv_use_fp32 = kv_dtype_env && std::strcmp(kv_dtype_env, "fp32") == 0;
    const bool kv_use_q8 = kv_dtype_env && std::strcmp(kv_dtype_env, "q8") == 0;
    const bool kv_use_fp8 = kv_dtype_env && std::strcmp(kv_dtype_env, "fp8") == 0;
    const bool kv_use_fp16 = !kv_use_fp32 && !kv_use_q8 && !kv_use_fp8;

    if (kv_use_q8) {
        mtp_k_cache_ = backend_.tensor_q8_kv(
            kv_per_pos * physical_slots, cfg.head_dim, "kvmem_bounded_mtp_k_cache");
        mtp_v_cache_ = backend_.tensor_q8_kv(
            kv_per_pos * physical_slots, cfg.head_dim, "kvmem_bounded_mtp_v_cache");
    } else if (kv_use_fp8) {
        mtp_k_cache_ = backend_.tensor_fp8_kv(
            kv_per_pos * physical_slots, "kvmem_bounded_mtp_k_cache");
        mtp_v_cache_ = backend_.tensor_fp8_kv(
            kv_per_pos * physical_slots, "kvmem_bounded_mtp_v_cache");
    } else if (kv_use_fp16) {
        mtp_k_cache_ = backend_.tensor_f16(
            kv_per_pos * physical_slots, "kvmem_bounded_mtp_k_cache");
        mtp_v_cache_ = backend_.tensor_f16(
            kv_per_pos * physical_slots, "kvmem_bounded_mtp_v_cache");
    } else {
        mtp_k_cache_ = backend_.tensor_f32(
            kv_per_pos * physical_slots, "kvmem_bounded_mtp_k_cache");
        mtp_v_cache_ = backend_.tensor_f32(
            kv_per_pos * physical_slots, "kvmem_bounded_mtp_v_cache");
    }
}

void QwenExecutor::ensure_mtp_scratch() {
    ensure_scratch();
    if (mtp_scratch_ready_) return;
    if (!weights_.mtp()) return;

    const QwenConfig &cfg = model_.config();
    mtp_h_ = backend_.tensor_f32(cfg.n_embd, "mtp_h");
    mtp_embd_ = backend_.tensor_f32(cfg.n_embd, "mtp_embd");
    mtp_enorm_ = backend_.tensor_f32(cfg.n_embd, "mtp_enorm");
    mtp_hnorm_ = backend_.tensor_f32(cfg.n_embd, "mtp_hnorm");
    mtp_concat_ = backend_.tensor_f32(static_cast<uint64_t>(2) * cfg.n_embd, "mtp_concat");
    mtp_zero_h_ = backend_.tensor_f32(cfg.n_embd, "mtp_zero_h");
    mtp_prefix_h_ = backend_.tensor_f32(cfg.n_embd, "mtp_prefix_h");
    (void) backend_.zero_tensor(*mtp_zero_h_);

    const uint64_t kv_per_pos = static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
    const uint64_t kv_slots = std::max<uint32_t>(kv_ctx_size_, 1);
    const char *kv_dtype_env = std::getenv("QW3_KV_DTYPE");
    const bool kv_use_fp32 = kv_dtype_env && std::strcmp(kv_dtype_env, "fp32") == 0;
    const bool kv_use_q8 = kv_dtype_env && std::strcmp(kv_dtype_env, "q8") == 0;
    const bool kv_use_fp8 = kv_dtype_env && std::strcmp(kv_dtype_env, "fp8") == 0;
    const bool kv_use_fp16 = !kv_use_fp32 && !kv_use_q8 && !kv_use_fp8;
    if (external_mtp_kv_cache_) {
        if (external_mtp_kv_cache_->k_cache.empty() ||
            external_mtp_kv_cache_->v_cache.empty() ||
            !external_mtp_kv_cache_->k_cache.front() ||
            !external_mtp_kv_cache_->v_cache.front()) {
            throw std::runtime_error("external MTP KV cache is incomplete");
        }
    } else if (kvmem_mtp_tiered_) {
        // Tiered MTP: mtp_k_cache_/mtp_v_cache_ were already sized to the
        // bounded pool by allocate_kvmem_mtp_gpu_cache (during configure_kvmem,
        // before any forward). Do NOT allocate the dense full-context cache.
        if (!mtp_k_cache_ || !mtp_v_cache_) {
            throw std::runtime_error(
                "kvmem_mtp_tiered_ set but bounded MTP GPU cache not allocated");
        }
    } else if (kv_use_q8) {
        mtp_k_cache_ = backend_.tensor_q8_kv(kv_per_pos * kv_slots, cfg.head_dim, "mtp_k_cache");
        mtp_v_cache_ = backend_.tensor_q8_kv(kv_per_pos * kv_slots, cfg.head_dim, "mtp_v_cache");
    } else if (kv_use_fp8) {
        mtp_k_cache_ = backend_.tensor_fp8_kv(kv_per_pos * kv_slots, "mtp_k_cache");
        mtp_v_cache_ = backend_.tensor_fp8_kv(kv_per_pos * kv_slots, "mtp_v_cache");
    } else if (kv_use_fp16) {
        mtp_k_cache_ = backend_.tensor_f16(kv_per_pos * kv_slots, "mtp_k_cache");
        mtp_v_cache_ = backend_.tensor_f16(kv_per_pos * kv_slots, "mtp_v_cache");
    } else {
        mtp_k_cache_ = backend_.tensor_f32(kv_per_pos * kv_slots, "mtp_k_cache");
        mtp_v_cache_ = backend_.tensor_f32(kv_per_pos * kv_slots, "mtp_v_cache");
    }
    mtp_scratch_ready_ = true;
}

void QwenExecutor::ensure_mtp_batch_scratch(uint32_t batch) {
    ensure_mtp_scratch();
    if (batch == 0 || batch <= mtp_batch_capacity_) return;
    const QwenMtpWeights *mtp = weights_.mtp();
    if (!mtp) return;
    const QwenConfig &cfg = model_.config();
    const QwenLayerWeights &layer = mtp->layer;
    const uint64_t B = batch;
    mtp_h_input_batch_ = backend_.scratch_f32(B * cfg.n_embd, "mtp_h_input_batch");
    mtp_h_batch_ = backend_.scratch_f32(B * cfg.n_embd, "mtp_h_batch");
    mtp_norm_batch_ = backend_.scratch_f32(B * cfg.n_embd, "mtp_norm_batch");
    mtp_concat_batch_ = backend_.scratch_f32(B * static_cast<uint64_t>(2) * cfg.n_embd,
                                             "mtp_concat_batch");
    mtp_q_batch_ = backend_.scratch_f32(B * std::max<uint64_t>(layer.q_rows, 1), "mtp_q_batch");
    mtp_k_batch_ = backend_.scratch_f32(B * std::max<uint64_t>(layer.k_rows, 1), "mtp_k_batch");
    mtp_v_batch_ = backend_.scratch_f32(B * std::max<uint64_t>(layer.v_rows, 1), "mtp_v_batch");
    mtp_mid_batch_ = backend_.scratch_f32(B * static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim,
                                          "mtp_mid_batch");
    mtp_ffn_gate_batch_ = backend_.scratch_f32(B * std::max<uint64_t>(layer.ffn_dim, 1),
                                               "mtp_ffn_gate_batch");
    mtp_ffn_up_batch_ = backend_.scratch_f32(B * std::max<uint64_t>(layer.ffn_dim, 1),
                                             "mtp_ffn_up_batch");
    mtp_ffn_mid_batch_ = backend_.scratch_f32(B * std::max<uint64_t>(layer.ffn_dim, 1),
                                              "mtp_ffn_mid_batch");
    mtp_ffn_out_batch_ = backend_.scratch_f32(B * cfg.n_embd, "mtp_ffn_out_batch");
    mtp_batch_capacity_ = batch;
}

void QwenExecutor::ensure_logits_batch_scratch(uint32_t batch) {
    if (batch == 0 || batch <= logits_batch_capacity_) return;
    const uint64_t vocab = weights_.output().rows;
    logits_batch_ = backend_.scratch_f32(static_cast<uint64_t>(batch) * vocab,
                                         "logits_batch");
    logits_batch_capacity_ = batch;
}

void QwenExecutor::ensure_batch_scratch(uint32_t batch) {
    if (batch == 0) return;
    if (batch <= batch_capacity_) return;
    const QwenConfig &cfg = model_.config();

    // Determine worst-case per-layer dims (mirrors ensure_scratch).
    uint64_t max_ffn = 0, max_q = 0, max_k = 0, max_v = 0;
    uint64_t max_rqkv = 0, max_rvalue = 0;
    for (uint32_t i = 0; i < weights_.n_layers(); ++i) {
        const QwenLayerWeights &l = weights_.layer(i);
        if (l.ffn_dim > max_ffn) max_ffn = l.ffn_dim;
        if (l.q_rows > max_q) max_q = l.q_rows;
        if (l.k_rows > max_k) max_k = l.k_rows;
        if (l.v_rows > max_v) max_v = l.v_rows;
        if (l.recurrent_qkv_dim > max_rqkv) max_rqkv = l.recurrent_qkv_dim;
        if (l.recurrent_value_dim > max_rvalue) max_rvalue = l.recurrent_value_dim;
    }

    const uint64_t B = batch;
    h_batch_       = backend_.tensor_f32(B * cfg.n_embd,             "h_batch");
    norm_batch_    = backend_.tensor_f32(B * cfg.n_embd,             "norm_batch");
    attn_out_batch_= backend_.tensor_f32(B * cfg.n_embd,             "attn_out_batch");
    ffn_gate_batch_= backend_.tensor_f32(B * std::max<uint64_t>(max_ffn, 1), "ffn_gate_batch");
    ffn_up_batch_  = backend_.tensor_f32(B * std::max<uint64_t>(max_ffn, 1), "ffn_up_batch");
    ffn_mid_batch_ = backend_.tensor_f32(B * std::max<uint64_t>(max_ffn, 1), "ffn_mid_batch");
    ffn_out_batch_ = backend_.tensor_f32(B * cfg.n_embd,             "ffn_out_batch");
    if (max_rqkv  > 0) proj_batch_      = backend_.tensor_f32(B * max_rqkv,  "proj_batch");
    if (max_rqkv  > 0) conv_out_batch_  = backend_.tensor_f32(B * max_rqkv,  "conv_out_batch");
    if (max_rvalue> 0) gate_proj_batch_ = backend_.tensor_f32(B * max_rvalue,"gate_proj_batch");
    if (max_rvalue> 0) core_batch_      = backend_.tensor_f32(B * max_rvalue,"core_batch");
    if (cfg.num_v_heads() > 0) {
        alpha_batch_ = backend_.tensor_f32(B * cfg.num_v_heads(),    "alpha_batch");
        beta_batch_  = backend_.tensor_f32(B * cfg.num_v_heads(),    "beta_batch");
    }
    if (max_q > 0) q_batch_ = backend_.tensor_f32(B * max_q, "q_batch");
    if (max_k > 0) k_batch_ = backend_.tensor_f32(B * max_k, "k_batch");
    if (max_v > 0) v_batch_ = backend_.tensor_f32(B * max_v, "v_batch");
    mid_batch_ = backend_.tensor_f32(B * static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim, "mid_batch");

    batch_capacity_ = batch;
}

uint64_t QwenExecutor::per_token_scratch_bytes() const {
    const QwenConfig &cfg = model_.config();
    uint64_t max_ffn = 0, max_q = 0, max_k = 0, max_v = 0;
    uint64_t max_rqkv = 0, max_rvalue = 0;
    for (uint32_t i = 0; i < weights_.n_layers(); ++i) {
        const QwenLayerWeights &l = weights_.layer(i);
        if (l.ffn_dim > max_ffn) max_ffn = l.ffn_dim;
        if (l.q_rows > max_q) max_q = l.q_rows;
        if (l.k_rows > max_k) max_k = l.k_rows;
        if (l.v_rows > max_v) max_v = l.v_rows;
        if (l.recurrent_qkv_dim > max_rqkv) max_rqkv = l.recurrent_qkv_dim;
        if (l.recurrent_value_dim > max_rvalue) max_rvalue = l.recurrent_value_dim;
    }
    uint64_t per_tok = 0;
    per_tok += 3 * cfg.n_embd;                                    // h, norm, attn_out
    per_tok += 3 * std::max<uint64_t>(max_ffn, 1);                // ffn_gate, ffn_up, ffn_mid
    per_tok += cfg.n_embd;                                        // ffn_out
    if (max_rqkv  > 0) per_tok += 2 * max_rqkv;                   // proj, conv_out
    if (max_rvalue> 0) per_tok += 2 * max_rvalue;                 // gate_proj, core
    if (cfg.num_v_heads() > 0) per_tok += 2 * cfg.num_v_heads();  // alpha, beta
    if (max_q > 0) per_tok += max_q;
    if (max_k > 0) per_tok += max_k;
    if (max_v > 0) per_tok += max_v;
    per_tok += static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim; // mid
    return per_tok * sizeof(float);
}

NativeExecutorReport QwenExecutor::dry_run_token(uint32_t token_id, bool execute_heavy) {
    if (execute_heavy) return forward_one_token(token_id);

    NativeExecutorReport report;
    const NativePlanInfo &plan = model_.plan();
    if (!plan.supported) {
        report.missing_kernels.push_back("native model plan is incomplete");
        return report;
    }
    ensure_scratch();

    require_status(backend_.begin());
    begin_record_timing(full_executor_trace_enabled());
    require_status(backend_.q8_0_get_row(*h_, weights_.token_embd(), token_id));
    record(report, "token_embedding_lookup");
    require_status(backend_.rms_norm(*norm_, *h_, weights_.output_norm(), model_.config().rms_eps));
    record(report, "output_norm");
    require_status(backend_.end());

    report.ok = true;
    return report;
}

NativeExecutorReport QwenExecutor::forward_one_token(uint32_t token_id,
                                                     bool compute_logits) {
    NativeExecutorReport report;
    const NativePlanInfo &plan = model_.plan();
    if (!plan.supported) {
        report.missing_kernels.push_back("native model plan is incomplete");
        return report;
    }
    ensure_scratch();

    const QwenConfig &cfg = model_.config();
    const uint32_t head_k_dim = cfg.head_k_dim();
    const uint32_t head_v_dim = cfg.head_v_dim_ssm();
    const uint32_t num_k_heads = cfg.num_k_heads();
    const uint32_t num_v_heads = cfg.num_v_heads();
    const uint32_t standard_head_dim = cfg.head_dim;
    const uint32_t standard_n_heads = cfg.n_heads;
    const uint32_t standard_n_kv_heads = cfg.n_kv_heads;
    const float eps = cfg.rms_eps;
    const bool kvmem_trace_this_token =
        compute_logits && kvmem_attn_trace_sample_now();
    const bool global_trace_this_token =
        compute_logits && global_attn_trace_sample_now();

    if (kvmem_immutable_source_k_) {
        const uint32_t bt = std::max<uint32_t>(
            1, block_store_ ? block_store_->config().block_tokens : 1);
        kvmem_ensure_raw_capture_capacity(bt);
    }
    require_status(backend_.begin());
    begin_record_timing(executor_trace_timing_enabled());
    if (kvmem_immutable_source_k_) {
        const uint32_t bt = std::max<uint32_t>(
            1, block_store_->config().block_tokens);
        const int64_t block_start =
            static_cast<int64_t>((position_ / bt) * bt);
        if (kvmem_raw_decode_block_start_ >= 0 &&
            kvmem_raw_decode_block_start_ != block_start) {
            kvmem_flush_raw_k_decode();
        }
        if (kvmem_raw_decode_block_start_ < 0) {
            kvmem_raw_decode_block_start_ = block_start;
            kvmem_raw_decode_first_row_ = position_ % bt;
            kvmem_raw_decode_rows_ = 0;
        }
    }

    // CUDA-graph capture path: skip on the first token (warm-up: every
    // backend-side scratch buffer needs to be sized before we record
    // pointers into a graph). Also disabled whenever we're inside an MTP
    // verify/replay pass (compute_logits == false) — the captured topology
    // assumes the full LM-head argmax tail runs, which the no-logits path
    // skips, so re-using a stale graph would be incorrect.
    // Paged KV currently uploads the host page table during append/attention;
    // keep decode eager until page tables are device-resident across steps.
    const bool try_capture = false;

    require_status(backend_.q8_0_get_row(*h_, weights_.token_embd(), token_id));
    record(report, "token_embedding_lookup");

    for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
        const QwenLayerWeights &layer = weights_.layer(il);
        require_status(backend_.rms_norm(*norm_, *h_, *layer.attn_norm, eps));
        record(report, "layer." + std::to_string(il) + ".attn_norm");

        if (layer.recurrent) {
            {
                DeviceTensor *outs[4] = {proj_.get(), gate_proj_.get(), alpha_.get(), beta_.get()};
                const DeviceWeight *ws[4] = {layer.attn_qkv, layer.attn_gate,
                                              layer.ssm_alpha, layer.ssm_beta};
                require_status(backend_.q8_0_matvec_fanout(outs, ws, 4, *norm_));
            }
            record(report, "layer." + std::to_string(il) + ".recurrent_projections");
            if (!recurrent_states_[il] || !conv_states_[il]) {
                throw std::runtime_error("recurrent state not allocated for layer " + std::to_string(il));
            }
            require_status(backend_.recurrent_single_token(*core_,
                                                           *recurrent_states_[il],
                                                           *conv_states_[il],
                                                           *conv_out_,
                                                           *proj_,
                                                           *gate_proj_,
                                                           *alpha_,
                                                           *beta_,
                                                           *layer.ssm_conv1d,
                                                           *layer.ssm_a,
                                                           *layer.ssm_dt_bias,
                                                           *layer.ssm_norm,
                                                           num_k_heads,
                                                           num_v_heads,
                                                           head_k_dim,
                                                           head_v_dim,
                                                           cfg.ssm_conv_kernel,
                                                           eps));
            record(report, "layer." + std::to_string(il) + ".deltanet_single_token");
            // Fused matvec + residual add: h += W_out * core. Falls back to
            // separate matvec + add inside the backend when the fused path
            // is unavailable (e.g. legacy QW3_MATVEC=qw3).
            if (auto st = backend_.q8_0_matvec_add(*h_, *layer.ssm_out, *core_); !st.ok) {
                require_status(backend_.q8_0_matvec(*attn_out_, *layer.ssm_out, *core_));
                require_status(backend_.add(*h_, *h_, *attn_out_));
            }
            record(report, "layer." + std::to_string(il) + ".recurrent_output_add");
        } else {
            ensure_kv_pages(position_, 1);
            // Block-sparse decode: the live KV cache (kv_pages_ @ position_) is
            // the growing repository, but attention runs over the assembled
            // WINDOW — the selected blocks' physical pages re-RoPE'd into a
            // contiguous [0..window_query_pos_) range. The new token's Q/K are
            // baked at the window tail, its KV appended at the window tail slot,
            // and attention scans the window page list. Under an identity (all-
            // block) selection window_* == kv_pages_/position_ exactly, so these
            // branches are byte-identical to the plain path below.
            const bool bs = kvmem_active_;
            if (bs) kvmem_extend_window_for_decode();
            const uint32_t attn_pos = bs ? window_query_pos_ : position_;
            // RoPE position: under the no-re-RoPE experiment the window keeps true
            // positions, so the new token's Q/K rotate at the TRUE position_ while
            // the append-slot and (length-based) attention mask stay on attn_pos.
            const uint32_t rope_pos =
                (bs && kvmem_no_rerope_) ? position_ : attn_pos;
            if (kvmem_mtp_local_positions_ && cfg.n_ctx_train > 0 &&
                rope_pos >= cfg.n_ctx_train) {
                throw std::runtime_error(
                    "KVMem target decode RoPE position exceeds the model "
                    "context limit in local-position MTP mode");
            }
            trace_rope_position_if_out_of_range(
                "forward_one_token.qk", rope_pos, 1, cfg.n_ctx_train,
                static_cast<int32_t>(il), /*kernel_uses=*/2);
            const DeviceTensor &pages_dev =
                bs ? *window_pages_device_ : kv_page_indices_device();
            const uint32_t pages_count = bs ? window_page_count_ : kv_page_count();
            {
                DeviceTensor *outs[3] = {q_.get(), k_.get(), v_.get()};
                const DeviceWeight *ws[3] = {layer.attn_q, layer.attn_k, layer.attn_v};
                require_status(backend_.q8_0_matvec_fanout(outs, ws, 3, *norm_));
            }
            record(report, "layer." + std::to_string(il) + ".attention_qkv_projection");

            // Per-head RMS norm using the shared head_dim-vector. Q is laid
            // out as [n_heads, 2, head_dim] so the per-unit stride is 2 *
            // head_dim and we normalize only the first head_dim (attn-Q).
            require_status(backend_.rmsnorm_per_head(*q_, *layer.attn_q_norm,
                                                     standard_n_heads,
                                                     2 * standard_head_dim,
                                                     standard_head_dim, eps));
            require_status(backend_.rmsnorm_per_head(*k_, *layer.attn_k_norm,
                                                     standard_n_kv_heads,
                                                     standard_head_dim,
                                                     standard_head_dim, eps));
            if (kvmem_immutable_source_k_) {
                kvmem_capture_raw_k_decode(
                    il, *k_, position_ % block_store_->config().block_tokens);
            }

            // Partial RoPE on the first rope_dim of each head's first segment.
            // Baked at the window position when block-sparse is active.
            require_status(backend_.rope_partial(*q_, standard_n_heads,
                                                 2 * standard_head_dim,
                                                 cfg.rope_dim, rope_pos, cfg.rope_theta));
            require_status(backend_.rope_partial(*k_, standard_n_kv_heads,
                                                 standard_head_dim,
                                                 cfg.rope_dim, rope_pos,
                                                 cfg.rope_theta));

            // Above-budget QC content index for the generated token: de-RoPE the
            // freshly-baked K at its bake position into the content frame (position-
            // invariant, so reselect re-baking the window is immaterial). No-op unless
            // decode capture was armed on the plain path (kvmem, above budget, mean-k).
            if (bs) kvmem_decode_capture_stage(il, rope_pos);

            // Append K and V to the live cache.
            const uint32_t per_pos = standard_n_kv_heads * standard_head_dim;
            require_status(backend_.kv_append_batch_paged_device(
                k_cache(il), *k_, attn_pos, per_pos, 1,
                pages_dev, pages_count, kv_page_size()));
            require_status(backend_.kv_append_batch_paged_device(
                v_cache(il), *v_, attn_pos, per_pos, 1,
                pages_dev, pages_count, kv_page_size()));
            record(report, "layer." + std::to_string(il) + ".kv_append_paged");

            const float scale = 1.0f / std::sqrt(static_cast<float>(standard_head_dim));
            require_status(backend_.attention_decode_batch_paged_gated_device(
                *mid_, *q_, 2 * standard_head_dim,
                attention_k_cache(il), v_cache(il),
                pages_dev, pages_count, kv_page_size(),
                standard_n_heads, standard_n_kv_heads, standard_head_dim,
                attn_pos, 1,
                standard_n_heads * 2 * standard_head_dim,
                standard_n_heads * standard_head_dim, scale));
            record(report, "layer." + std::to_string(il) + ".attention_sdpa_paged");

            if (global_trace_this_token) {
                global_trace_attention_layer(
                    il, attention_k_cache(il), *q_, 2 * standard_head_dim,
                    pages_dev, pages_count, attn_pos + 1, scale);
            }

            if (bs && kvmem_trace_this_token) {
                kvmem_trace_attention_layer(
                    il, attention_k_cache(il), *q_, 2 * standard_head_dim,
                    pages_dev, pages_count, attn_pos + 1, scale);
            }

            // Cumulative-attention selection signal (#40): score the current Q
            // against each window block's representative K at the representative
            // layer only. One extra global launch per step; no D2H. Inert unless
            // block-sparse is active and k̄ is live for this interval.
            if (bs) kvmem_score_current_step(il, scale);

            // Global content-frame retrieval (#49): de-RoPE the current Q into
            // the content frame at the representative layer so it can be scored
            // against the global content index at the next retrieval boundary.
            // Inert unless the global index is live (fp16/fp32 cache).
            if (bs) kvmem_snapshot_content_query(il);

            // Fused matvec + residual add: h += W_out * mid.
            if (auto st = backend_.q8_0_matvec_add(*h_, *layer.attn_output, *mid_); !st.ok) {
                require_status(backend_.q8_0_matvec(*attn_out_, *layer.attn_output, *mid_));
                require_status(backend_.add(*h_, *h_, *attn_out_));
            }
        }
        record(report, "layer." + std::to_string(il) + ".attn_residual");

        require_status(backend_.rms_norm(*norm_, *h_, *layer.ffn_norm, eps));
        record(report, "layer." + std::to_string(il) + ".ffn_norm");

        // Fused FFN SwiGLU: ffn_mid = silu(W_gate * norm) * (W_up * norm)
        // in a single matvec kernel. Falls back to the two-weight matvec +
        // silu_mul pipeline if the backend doesn't implement the fused op.
        if (auto st = backend_.q8_0_matvec_silu_mul(*ffn_mid_, *layer.ffn_gate,
                                                    *layer.ffn_up, *norm_);
            !st.ok) {
            DeviceTensor *outs[2] = {ffn_gate_.get(), ffn_up_.get()};
            const DeviceWeight *ws[2] = {layer.ffn_gate, layer.ffn_up};
            require_status(backend_.q8_0_matvec_fanout(outs, ws, 2, *norm_));
            require_status(backend_.silu_mul(*ffn_mid_, *ffn_gate_, *ffn_up_));
        }
        // Fused matvec + residual add: h += W_down * ffn_mid.
        if (auto st = backend_.q8_0_matvec_add(*h_, *layer.ffn_down, *ffn_mid_); !st.ok) {
            require_status(backend_.q8_0_matvec(*ffn_out_, *layer.ffn_down, *ffn_mid_));
            require_status(backend_.add(*h_, *h_, *ffn_out_));
        }
        record(report, "layer." + std::to_string(il) + ".ffn");
    }

    if (kvmem_immutable_source_k_) {
        const uint32_t bt = block_store_->config().block_tokens;
        const uint32_t row = position_ % bt;
        if (kvmem_raw_decode_rows_ == 0) {
            kvmem_raw_decode_first_row_ = row;
        }
        kvmem_raw_decode_rows_ =
            row - kvmem_raw_decode_first_row_ + 1;
        if (row + 1 == bt) kvmem_flush_raw_k_decode();
    }

    if (!compute_logits) {
        // MTP verify/replay re-runs the target model only to advance KV +
        // recurrent state; the LM-head argmax tail is skipped. Note: graph
        // capture is suppressed in this path (try_capture is false), so the
        // eager begin/end pairing here is always correct.
        require_status(backend_.end());
        position_++;
        if (kvmem_active_) window_query_pos_++;
        report.ok = true;
        return report;
    }

    require_status(backend_.rms_norm(*norm_, *h_, weights_.output_norm(), eps));
    record(report, "output_norm");
    require_status(backend_.q8_0_matvec(*logits_, weights_.output(), *norm_));

    DeviceArgmax best;
    if (try_capture) {
        // Record argmax kernel + its async D2H into the captured graph.
        require_status(backend_.argmax_launch(*logits_));
        require_status(backend_.end_capture());
        require_status(backend_.replay_graph());
        // Sync + read the pinned argmax mirror after the graph has run.
        best = backend_.argmax_collect();
    } else {
        // Eager path: pulls in the warm-up token and any token where
        // capture refused. Sets decode_graph_warmup_pending_ to false so
        // the next call attempts capture.
        best = backend_.argmax(*logits_);
        decode_graph_warmup_pending_ = false;
    }
    require_status(backend_.end());

    position_++;
    if (kvmem_active_) window_query_pos_++;
    if (kvmem_active_ && kvmem_attn_trace_enabled()) {
        ++kvmem_attn_trace_seen_tokens_;
    }
    if (kvmem_trace_this_token) ++kvmem_attn_trace_sample_;
    if (global_attn_trace_enabled()) {
        ++global_attn_trace_seen_tokens_;
    }
    if (global_trace_this_token) ++global_attn_trace_sample_;
    report.argmax_token = best.token;
    report.argmax_logit = best.logit;
    report.argmax_text = model_.gguf().token_text(static_cast<uint32_t>(best.token));
    record(report, "lm_head_argmax");
    report.ok = true;
    return report;
}

NativeExecutorReport QwenExecutor::forward_recurrent_layer_from_current_hidden(
        uint32_t layer_index) {
    NativeExecutorReport report;
    const NativePlanInfo &plan = model_.plan();
    if (!plan.supported) {
        report.missing_kernels.push_back("native model plan is incomplete");
        return report;
    }
    if (layer_index >= weights_.n_layers()) {
        report.missing_kernels.push_back("recurrent layer index out of range");
        return report;
    }
    ensure_scratch();

    const QwenConfig &cfg = model_.config();
    const QwenLayerWeights &layer = weights_.layer(layer_index);
    if (!layer.recurrent) {
        report.missing_kernels.push_back("requested layer is not recurrent");
        return report;
    }
    const uint32_t head_k_dim = cfg.head_k_dim();
    const uint32_t head_v_dim = cfg.head_v_dim_ssm();
    const uint32_t num_k_heads = cfg.num_k_heads();
    const uint32_t num_v_heads = cfg.num_v_heads();
    const float eps = cfg.rms_eps;

    require_status(backend_.begin());
    begin_record_timing(executor_trace_timing_enabled());

    require_status(backend_.rms_norm(*norm_, *h_, *layer.attn_norm, eps));
    record(report, "layer." + std::to_string(layer_index) + ".attn_norm");
    {
        DeviceTensor *outs[4] = {proj_.get(), gate_proj_.get(), alpha_.get(), beta_.get()};
        const DeviceWeight *ws[4] = {layer.attn_qkv, layer.attn_gate,
                                      layer.ssm_alpha, layer.ssm_beta};
        require_status(backend_.q8_0_matvec_fanout(outs, ws, 4, *norm_));
    }
    record(report, "layer." + std::to_string(layer_index) + ".recurrent_projections");
    if (!recurrent_states_[layer_index] || !conv_states_[layer_index]) {
        throw std::runtime_error("recurrent state not allocated for layer " +
                                 std::to_string(layer_index));
    }
    require_status(backend_.recurrent_single_token(*core_,
                                                   *recurrent_states_[layer_index],
                                                   *conv_states_[layer_index],
                                                   *conv_out_,
                                                   *proj_,
                                                   *gate_proj_,
                                                   *alpha_,
                                                   *beta_,
                                                   *layer.ssm_conv1d,
                                                   *layer.ssm_a,
                                                   *layer.ssm_dt_bias,
                                                   *layer.ssm_norm,
                                                   num_k_heads,
                                                   num_v_heads,
                                                   head_k_dim,
                                                   head_v_dim,
                                                   cfg.ssm_conv_kernel,
                                                   eps));
    record(report, "layer." + std::to_string(layer_index) + ".deltanet_single_token");
    if (auto st = backend_.q8_0_matvec_add(*h_, *layer.ssm_out, *core_); !st.ok) {
        require_status(backend_.q8_0_matvec(*attn_out_, *layer.ssm_out, *core_));
        require_status(backend_.add(*h_, *h_, *attn_out_));
    }
    record(report, "layer." + std::to_string(layer_index) + ".recurrent_output_add");
    record(report, "layer." + std::to_string(layer_index) + ".attn_residual");

    require_status(backend_.rms_norm(*norm_, *h_, *layer.ffn_norm, eps));
    record(report, "layer." + std::to_string(layer_index) + ".ffn_norm");
    if (auto st = backend_.q8_0_matvec_silu_mul(*ffn_mid_, *layer.ffn_gate,
                                                *layer.ffn_up, *norm_);
        !st.ok) {
        DeviceTensor *outs[2] = {ffn_gate_.get(), ffn_up_.get()};
        const DeviceWeight *ws[2] = {layer.ffn_gate, layer.ffn_up};
        require_status(backend_.q8_0_matvec_fanout(outs, ws, 2, *norm_));
        require_status(backend_.silu_mul(*ffn_mid_, *ffn_gate_, *ffn_up_));
    }
    if (auto st = backend_.q8_0_matvec_add(*h_, *layer.ffn_down, *ffn_mid_); !st.ok) {
        require_status(backend_.q8_0_matvec(*ffn_out_, *layer.ffn_down, *ffn_mid_));
        require_status(backend_.add(*h_, *h_, *ffn_out_));
    }
    record(report, "layer." + std::to_string(layer_index) + ".ffn");

    require_status(backend_.end());
    report.ok = true;
    return report;
}

uint32_t QwenExecutor::effective_prefill_chunk_size(uint32_t total) const {
    if (total == 0) return 0;
    // Default cap 2048: recovers most of the chunking throughput tax vs
    // whole-prompt while holding peak scratch close to chunk=512. See
    // forward_n_tokens for the tuning rationale.
    constexpr uint32_t kQw3DefaultPrefillChunk = 2048;
    uint32_t chunk_size = std::min<uint32_t>(kQw3DefaultPrefillChunk, total);
    // CLI override (`--prefill-chunk N`) takes precedence over env. -1 means
    // "no override, use env or default".
    if (prefill_chunk_override_ >= 0) {
        if (prefill_chunk_override_ == 0) {
            chunk_size = total;  // whole-prompt batch
        } else {
            chunk_size = static_cast<uint32_t>(prefill_chunk_override_);
        }
    } else if (const char *env = std::getenv("QW3_PREFILL_CHUNK")) {
        int v = std::atoi(env);
        if (v > 0) {
            chunk_size = static_cast<uint32_t>(v);
        } else if (v == 0) {
            // Explicit opt-out of chunking.
            chunk_size = total;
        }
    }
    // Safety floor: even if the user set a large chunk (or QW3_PREFILL_CHUNK=0
    // disabled the cap), don't exceed what fits in 80% of currently free
    // device memory. This handles edge cases where weights + KV cache leave
    // less headroom than the requested chunk's per-prompt scratch.
    const uint64_t per_tok = per_token_scratch_bytes();
    if (per_tok > 0) {
        const uint64_t free_b = backend_.free_device_bytes();
        if (free_b > 0) {
            const uint64_t budget = (free_b * 8) / 10;
            const uint64_t fits = budget / per_tok;
            if (fits > 0 && fits < chunk_size) {
                chunk_size = static_cast<uint32_t>(fits);
                if (chunk_size > 256) chunk_size &= ~static_cast<uint32_t>(255);
            }
        }
    }
    if (kvmem_gpu_page_pool_ && block_store_) {
        // The bounded GPU pool used to force prefill into block-granular
        // (block_tokens, e.g. 16-token) chunks. That collapsed every matmul
        // (attn projections, FFN, DeltaNet) into the tiny-batch MMVQ regime —
        // each weight tile is read but feeds only ~16 columns, so weight-read
        // amortization is ~chunk/block worse. Empirically this was a ~7x
        // prefill cliff the instant the pool engaged (the whole spill regime).
        //
        // The append for a chunk grabs ceil(batch/page_size) pages in one shot
        // before the post-chunk offload runs, so the only real constraint is
        // that a chunk must fit in the pool headroom left over the resident
        // window. After a reselect the window holds ~budget_blocks (+ recent
        // cushion); the rest of the pool is free for the next chunk's append.
        // Cap to that headroom (keeping one block of slack so the offload's own
        // keep-free cushion is honored), aligned down to a block boundary, and
        // never below block_tokens. The existing kvmem_maybe_prefill_offload
        // headroom reservation (next_chunk_pages + cushion) then keeps the pool
        // from overflowing for any chunk this cap allows.
        const uint32_t block_tokens =
            std::max<uint32_t>(1, block_store_->config().block_tokens);
        const uint32_t psz = std::max<uint32_t>(1, kv_pages_.page_size);
        const uint32_t pages_per_block =
            std::max<uint32_t>(1, block_tokens / psz);
        const uint32_t pool_pages = kvmem_gpu_page_pool_->total_pages();
        const uint32_t budget_blocks =
            std::max<uint32_t>(1, block_store_->budget_blocks());
        const uint32_t cushion_blocks =
            std::max<uint32_t>(1, block_store_->config().recent_blocks);
        const uint64_t resident_pages =
            static_cast<uint64_t>(budget_blocks + cushion_blocks) *
            pages_per_block;
        const uint32_t headroom_pages =
            static_cast<uint64_t>(pool_pages) > resident_pages
                ? static_cast<uint32_t>(pool_pages - resident_pages)
                : 0;
        // Use only a quarter of the budget->pool gap for a single chunk. A
        // chunk's pages are claimed in one shot before the post-chunk offload
        // runs, and the offload's reselect can transiently hold extra pages
        // (stage-in of any resurrected block before the stage-out release).
        // A quarter leaves generous margin on tight pools (where budget ~ pool)
        // while still letting the chunk reach the full default whenever the pool
        // is sized well above the window budget (the common case).
        const uint32_t chunk_pages_cap = headroom_pages / 4;
        uint32_t max_chunk_tokens =
            chunk_pages_cap > 0 ? chunk_pages_cap * psz : block_tokens;
        if (max_chunk_tokens < block_tokens) max_chunk_tokens = block_tokens;
        if (max_chunk_tokens > block_tokens) {
            max_chunk_tokens -= max_chunk_tokens % block_tokens;
        }
        chunk_size = std::min<uint32_t>(chunk_size, max_chunk_tokens);
    }
    if (chunk_size > total) chunk_size = total;
    if (chunk_size == 0) chunk_size = total;
    return chunk_size;
}

NativeExecutorReport QwenExecutor::forward_n_tokens(const std::vector<uint32_t> &tokens,
                                                    bool compute_logits,
                                                    std::vector<DeviceArgmax> *row_argmaxes,
                                                    StateCheckpointSet *state_checkpoints,
                                                    uint32_t state_checkpoint_count,
                                                    bool copy_last_logits,
                                                    std::vector<std::vector<float>> *row_logits_host) {
    NativeExecutorReport report;
    if (tokens.empty()) { report.ok = true; return report; }
    const NativePlanInfo &plan = model_.plan();
    if (!plan.supported) {
        report.missing_kernels.push_back("native model plan is incomplete");
        return report;
    }
    ensure_scratch();
    const uint32_t total = static_cast<uint32_t>(tokens.size());

    // Block-sparse (kvmem) batched verify: when a window is active this batch
    // appends + attends in the WINDOW frame (window_query_pos_ base + window
    // page table), exactly mirroring the single-token decode path
    // (forward_one_token bs branch). MTP verify is always a single chunk
    // (mtp_single_chunk forces chunk_size=total below), so chunk_off is always 0
    // and the window base position is window_query_pos_. Under identity
    // (all-block) selection window_query_pos_ == position_, so this stays
    // byte-identical to the plain path. Per-step attention heat
    // (kvmem_score_current_step / kvmem_snapshot_content_query) is NOT
    // accumulated here: the default Retrieval selector is position-invariant and
    // unaffected; H2O/Recency see slightly staler heat across an MTP verify
    // batch (v1 limitation).
    const bool bs_at_entry = kvmem_active_;

    // MTP verify/replay requires the whole batch to live in h_batch_ at the
    // tail (per-row argmax) and consistent checkpoint base positions, so it
    // must run as a single chunk. The verifier batch is tiny (chain length,
    // typically <= 8), so this never grows peak memory.
    const bool mtp_single_chunk = (row_argmaxes != nullptr) || (state_checkpoints != nullptr);

    // Prefill chunking. The chunk cap controls peak compute scratch (per-token
    // batch tensors: residual, FFN gate/up, q/k/v projections, etc.). qw3
    // originally sized batch scratch to the entire prompt, which made peak
    // memory grow linearly with T — at T=64K the per-prompt batch scratch
    // alone exceeded 30 GiB of FP32 storage. Capping the chunk holds peak
    // memory roughly flat in T.
    //
    // The cap is 2048: empirically this recovers most of the chunking
    // throughput tax (vs whole-prompt) while keeping peak memory close to
    // chunk=512 (within ~1.1 GiB at T=65K). Smaller chunks pay a per-chunk
    // amortization tax (HGEMM autotuner restart, MMQ-at-short-batch dispatch,
    // sub-saturation grids) without buying meaningful memory back; larger
    // chunks (≥4096) re-grow the per-chunk scratch significantly.
    //
    // Override with QW3_PREFILL_CHUNK=N. Set N=0 to disable the cap entirely
    // (whole-prompt batch — original behavior, useful for benchmarking the
    // throughput tax of chunking itself). effective_prefill_chunk_size() is the
    // single source of truth for the cap (override/env, free-memory floor, and
    // the bounded KVMem GPU pool headroom cap); the MTP prefix priming loop
    // queries the same function so its outer chunks match what runs here.
    uint32_t chunk_size = effective_prefill_chunk_size(total);
    if (mtp_single_chunk) chunk_size = total;  // MTP verify: never split
    if (chunk_size > total) chunk_size = total;
    if (chunk_size == 0) chunk_size = total;
    ensure_batch_scratch(chunk_size);
    if (kvmem_immutable_source_k_) {
        kvmem_ensure_raw_capture_capacity(chunk_size);
    }

    const QwenConfig &cfg = model_.config();
    const uint32_t head_k_dim = cfg.head_k_dim();
    const uint32_t head_v_dim = cfg.head_v_dim_ssm();
    const uint32_t num_k_heads = cfg.num_k_heads();
    const uint32_t num_v_heads = cfg.num_v_heads();
    const uint32_t standard_head_dim = cfg.head_dim;
    const uint32_t standard_n_heads = cfg.n_heads;
    const uint32_t standard_n_kv_heads = cfg.n_kv_heads;
    const float eps = cfg.rms_eps;

    auto row_stride = [this](const DeviceTensor *t) -> uint32_t {
        return static_cast<uint32_t>(t->count / batch_capacity_);
    };
    const uint32_t h_stride = row_stride(h_batch_.get());
    const uint32_t ffn_stride = row_stride(ffn_gate_batch_.get());
    const uint32_t q_stride_buf = q_batch_ ? row_stride(q_batch_.get()) : 0;
    const uint32_t k_stride_buf = k_batch_ ? row_stride(k_batch_.get()) : 0;
    const uint32_t v_stride_buf = v_batch_ ? row_stride(v_batch_.get()) : 0;
    const uint32_t mid_stride = row_stride(mid_batch_.get());
    const uint32_t proj_stride = proj_batch_ ? row_stride(proj_batch_.get()) : 0;
    const uint32_t gate_proj_stride = gate_proj_batch_ ? row_stride(gate_proj_batch_.get()) : 0;
    const uint32_t alpha_stride = alpha_batch_ ? row_stride(alpha_batch_.get()) : 0;
    const uint32_t beta_stride = beta_batch_ ? row_stride(beta_batch_.get()) : 0;
    const uint32_t core_stride = core_batch_ ? row_stride(core_batch_.get()) : 0;
    const bool use_paged_prefill =
        has_external_kv_cache() || paged_kv_prefill_for_local_cache_enabled();

    require_status(backend_.begin());
    begin_record_timing(full_executor_trace_enabled());
    if (kvmem_immutable_source_k_) kvmem_flush_raw_k_decode();

    // MTP rollback support: when state_checkpoints is requested, the recurrent
    // layers below snapshot their per-token DeltaNet + conv state into the
    // checkpoint slots so the orchestrator can roll back to any accepted
    // verifier row. Only meaningful in the single-chunk MTP path.
    const uint32_t save_state_checkpoints =
        state_checkpoints ? std::min<uint32_t>(state_checkpoint_count, total) : 0;
    if (state_checkpoints) {
        state_checkpoints->ready = false;
        state_checkpoints->base_position = position_;
        state_checkpoints->count = save_state_checkpoints;
        state_checkpoints->h_stride = h_stride;
        // Record the window base so restore_state_checkpoint can roll the window
        // tail back per accepted row (see StateCheckpointSet::kvmem_active). The
        // window is extended below (kvmem_extend_window_for_decode_n); its base
        // query pos / page count here are the pre-batch values.
        state_checkpoints->kvmem_active = bs_at_entry;
        state_checkpoints->window_base_query_pos = window_query_pos_;
        state_checkpoints->window_base_page_count = window_page_count_;
        if (state_checkpoints->recurrent_states.size() != recurrent_states_.size()) {
            state_checkpoints->recurrent_states.resize(recurrent_states_.size());
        }
        if (state_checkpoints->conv_states.size() != conv_states_.size()) {
            state_checkpoints->conv_states.resize(conv_states_.size());
        }
    }

    // Per-chunk graph capture is disabled while paged KV copies host page
    // metadata inside append/attention calls. Re-enable once page tables are
    // owned as stable device buffers by the scheduler.
    const bool prefill_graph_enabled = false;

    // Skip capture on the first chunk (warmup): backend-side scratch
    // (q8_1 staging, fattn workspace) sizes itself on first use. After
    // that, full-size chunks attempt capture+replay; the trailing partial
    // chunk always runs eagerly so we keep one stable graph topology.
    bool capture_warmup_pending = true;

    uint32_t last_chunk_batch = 0;
    last_forward_rows_ = 0;
    for (uint32_t chunk_off = 0; chunk_off < total; chunk_off += chunk_size) {
        const uint32_t batch = std::min(chunk_size, total - chunk_off);
        last_chunk_batch = batch;
        const uint32_t base_pos = position_ + chunk_off;
        const bool record_ops = (chunk_off == 0);
        const bool full_chunk = (batch == chunk_size);

        ensure_kv_pages(base_pos, batch);

        // Block-sparse: extend the assembled window so the batch's `batch`
        // tokens can be appended at window slots [window_query_pos_,
        // window_query_pos_+batch). The window's trailing pages alias the SAME
        // physical pages ensure_kv_pages just allocated at the true positions
        // [base_pos, base_pos+batch) (no copy). Append + attention below then
        // run in the WINDOW frame: base position window_query_pos_, window page
        // table. Under identity (all-block) selection window_query_pos_ ==
        // base_pos == position_ and the window page list == kv_pages_, so this
        // is byte-identical to the plain paged path. Non-bs: attn_base_pos ==
        // base_pos and the plain page table is used.
        const bool chunk_bs = kvmem_active_;
        if (chunk_bs) kvmem_extend_window_for_decode_n(batch, base_pos);
        const uint32_t attn_base_pos = chunk_bs ? window_query_pos_ : base_pos;
        // RoPE position for this chunk's Q/K (and the matching capture de-RoPE):
        // under the no-re-RoPE experiment use the TRUE base_pos so the window keeps
        // true positions; append-slot + (length-based) attention mask stay on
        // attn_base_pos. Default OFF -> rope_base_pos == attn_base_pos (unchanged).
        const uint32_t rope_base_pos =
            (chunk_bs && kvmem_no_rerope_) ? base_pos : attn_base_pos;
        last_forward_logical_base_ = base_pos;
        last_forward_rope_base_ = rope_base_pos;
        last_forward_rows_ = batch;
        const DeviceTensor &attn_pages_dev =
            chunk_bs ? *window_pages_device_ : kv_page_indices_device();
        const uint32_t attn_pages_count =
            chunk_bs ? window_page_count_ : kv_page_count();

        // Embedding lookup runs eagerly: q8_0_get_rows_batch issues a
        // pageable host->device memcpy which is unsafe inside stream capture.
        std::vector<uint64_t> rows_h(batch);
        for (uint32_t i = 0; i < batch; ++i) rows_h[i] = tokens[chunk_off + i];
        require_status(backend_.q8_0_get_rows_batch(*h_batch_, weights_.token_embd(), rows_h.data(), batch));
        if (record_ops) record(report, "token_embedding_lookup_batch");

        const bool try_capture = prefill_graph_enabled
            && full_chunk && !capture_warmup_pending && backend_.begin_capture();

        for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
        const QwenLayerWeights &layer = weights_.layer(il);
        require_status(backend_.rms_norm_batch(*norm_batch_, *h_batch_, *layer.attn_norm,
                                                batch, h_stride, eps));
        if (record_ops) record(report, "layer." + std::to_string(il) + ".attn_norm_batch");

        if (layer.recurrent) {
            {
                DeviceTensor *outs[4] = {proj_batch_.get(), gate_proj_batch_.get(),
                                         alpha_batch_.get(), beta_batch_.get()};
                const DeviceWeight *ws[4] = {layer.attn_qkv, layer.attn_gate,
                                             layer.ssm_alpha, layer.ssm_beta};
                const uint32_t strides[4] = {proj_stride, gate_proj_stride,
                                             alpha_stride, beta_stride};
                require_status(backend_.q8_0_matmul_fanout(outs, ws, strides, 4,
                                                           *norm_batch_, batch, h_stride));
            }
            if (record_ops) record(report, "layer." + std::to_string(il) + ".recurrent_projections_batch");
            if (!recurrent_states_[il] || !conv_states_[il] || !conv_out_batch_) {
                throw std::runtime_error("recurrent state not allocated for layer " + std::to_string(il));
            }
            DeviceTensor *state_checkpoint = nullptr;
            DeviceTensor *conv_state_checkpoint = nullptr;
            if (save_state_checkpoints > 0) {
                const uint64_t recurrent_count =
                    recurrent_states_[il]->count * save_state_checkpoints;
                if (!state_checkpoints->recurrent_states[il] ||
                    state_checkpoints->recurrent_states[il]->count != recurrent_count) {
                    state_checkpoints->recurrent_states[il] =
                        backend_.scratch_f32(recurrent_count,
                                             "mtp_state_checkpoint_recurrent");
                }
                const uint64_t conv_count =
                    conv_states_[il]->count * save_state_checkpoints;
                if (!state_checkpoints->conv_states[il] ||
                    state_checkpoints->conv_states[il]->count != conv_count) {
                    state_checkpoints->conv_states[il] =
                        backend_.scratch_f32(conv_count,
                                             "mtp_state_checkpoint_conv");
                }
                state_checkpoint = state_checkpoints->recurrent_states[il].get();
                conv_state_checkpoint = state_checkpoints->conv_states[il].get();
            }
            // DeltaNet-state retrieval capture (retrieval_method==deltanet;
            // deltanet_retrieval.md). For a selected DeltaNet layer with an active
            // question span, snapshot S_j per block + accumulate the in-block
            // log-decay so the reselect boundary can score blocks by their state
            // edits. Inert (nullptr) otherwise -> byte-identical to the base path.
            DeviceTensor *dn_snap = nullptr;
            DeviceTensor *dn_decay = nullptr;
            uint64_t dn_snap_off = 0, dn_decay_off = 0;
            uint32_t dn_bt = 0, dn_nblocks = 0;
            int32_t dn_slot_il = -1;
            if (kvmem_qc_deltanet_ && kvmem_query_end_ > kvmem_query_begin_ &&
                g_deltanet_snap_ && !kvmem_dn_ready_ &&
                static_cast<size_t>(il) < dn_layer_slot_.size()) {
                dn_slot_il = dn_layer_slot_[il];
            }
            if (dn_slot_il >= 0) {
                const uint32_t bt =
                    std::max<uint32_t>(block_store_->config().block_tokens, 1u);
                const uint32_t nblk = kvmem_qc_total_blocks_;
                const uint64_t snap_per_layer =
                    static_cast<uint64_t>(nblk) * num_v_heads * head_v_dim * head_k_dim;
                const uint64_t decay_per_layer =
                    static_cast<uint64_t>(nblk) * num_v_heads;
                dn_snap = g_deltanet_snap_.get();
                dn_decay = g_deltanet_decaysum_.get();
                dn_snap_off = static_cast<uint64_t>(dn_slot_il) * snap_per_layer;
                dn_decay_off = static_cast<uint64_t>(dn_slot_il) * decay_per_layer;
                dn_bt = bt;
                dn_nblocks = nblk;
            }
            // One batched call replaces the previous T-token loop (5 kernels x
            // T tokens). The CUDA backend overrides this with 4 launches per
            // layer that internally iterate over T.
            require_status(backend_.recurrent_batch(*core_batch_,
                                                     *recurrent_states_[il],
                                                     *conv_states_[il],
                                                     *conv_out_batch_,
                                                     *proj_batch_,
                                                     *gate_proj_batch_,
                                                     *alpha_batch_,
                                                     *beta_batch_,
                                                     *layer.ssm_conv1d,
                                                     *layer.ssm_a,
                                                     *layer.ssm_dt_bias,
                                                     *layer.ssm_norm,
                                                     batch,
                                                     num_k_heads,
                                                     num_v_heads,
                                                     head_k_dim,
                                                     head_v_dim,
                                                     cfg.ssm_conv_kernel,
                                                     layer.recurrent_qkv_dim,
                                                     proj_stride,
                                                     gate_proj_stride,
                                                     alpha_stride,
                                                     beta_stride,
                                                     core_stride,
                                                     eps,
                                                     state_checkpoint,
                                                     conv_state_checkpoint,
                                                     save_state_checkpoints,
                                                     dn_snap,
                                                     dn_decay,
                                                     dn_snap_off,
                                                     dn_decay_off,
                                                     dn_bt,
                                                     base_pos,
                                                     dn_nblocks));
            if (record_ops) record(report, "layer." + std::to_string(il) + ".deltanet_batch");
            // After recurrent_batch, conv_out_batch_ holds this layer's L2-normed
            // DeltaNet Q (first num_k_heads*head_k_dim of each row). Capture the
            // in-span question rows for DeltaNet-state scoring before the next
            // recurrent layer overwrites the buffer.
            if (dn_slot_il >= 0) {
                kvmem_capture_deltanet_query(static_cast<uint32_t>(dn_slot_il),
                                             chunk_off, batch, base_pos,
                                             *conv_out_batch_, proj_stride);
            }
            require_status(backend_.q8_0_matmul(*attn_out_batch_, *layer.ssm_out, *core_batch_,
                                                 batch, core_stride, h_stride));
            if (record_ops) record(report, "layer." + std::to_string(il) + ".recurrent_output_batch");
        } else {
            if (kvmem_mtp_local_positions_ && cfg.n_ctx_train > 0 &&
                static_cast<uint64_t>(rope_base_pos) + batch >
                    cfg.n_ctx_train) {
                throw std::runtime_error(
                    "KVMem target prefill/verify RoPE range exceeds the model "
                    "context limit in local-position MTP mode");
            }
            trace_rope_position_if_out_of_range(
                "forward_n_tokens.qk", rope_base_pos, batch,
                cfg.n_ctx_train, static_cast<int32_t>(il),
                /*kernel_uses=*/2);
            {
                DeviceTensor *outs[3] = {q_batch_.get(), k_batch_.get(), v_batch_.get()};
                const DeviceWeight *ws[3] = {layer.attn_q, layer.attn_k, layer.attn_v};
                const uint32_t strides[3] = {q_stride_buf, k_stride_buf, v_stride_buf};
                require_status(backend_.q8_0_matmul_fanout(outs, ws, strides, 3,
                                                           *norm_batch_, batch, h_stride));
            }
            if (record_ops) record(report, "layer." + std::to_string(il) + ".attention_qkv_projection_batch");

            require_status(backend_.rmsnorm_per_head_batch(*q_batch_, *layer.attn_q_norm,
                                                            batch, q_stride_buf,
                                                            standard_n_heads,
                                                            2 * standard_head_dim,
                                                            standard_head_dim, eps));
            require_status(backend_.rmsnorm_per_head_batch(*k_batch_, *layer.attn_k_norm,
                                                            batch, k_stride_buf,
                                                            standard_n_kv_heads,
                                                            standard_head_dim,
                                                            standard_head_dim, eps));
            if (kvmem_immutable_source_k_) {
                kvmem_capture_raw_k_batch(il, *k_batch_, batch);
            }

            require_status(backend_.rope_partial_batch(*q_batch_,
                                                        batch, q_stride_buf,
                                                        standard_n_heads,
                                                        2 * standard_head_dim,
                                                        cfg.rope_dim, rope_base_pos, cfg.rope_theta));
            require_status(backend_.rope_partial_batch(*k_batch_,
                                                        batch, k_stride_buf,
                                                        standard_n_kv_heads,
                                                        standard_head_dim,
                                                        cfg.rope_dim, rope_base_pos,
                                                        cfg.rope_theta));

            // Query-conditioned KVMem (#80/#87): capture the in-span question Q
            // rows during prefill and de-RoPE into the content frame for boundary
            // scoring. For long prompts (history >> budget) the GPU page pool
            // offloads mid-prefill, so kvmem_active_ (chunk_bs) is already true by
            // the time the trailing question tokens are prefilled — the question Q
            // is then RoPE'd at the WINDOW position (attn_base_pos = window_query_pos_),
            // not the true bake position. De-RoPE at whatever rotation was actually
            // applied (attn_base_pos, which equals base_pos when not window-active),
            // so the captured query lands in the same position-invariant content
            // frame as the k̄ index regardless of window state. Span membership still
            // keys off the TRUE prompt index (base_pos). Multi-layer (#87): fire at
            // EVERY normal-attention layer with its slot so block scoring folds in
            // all L layers; single-layer mode resolves to just bs_score_layer_
            // (slot 0). Inert unless a span is active.
            if (kvmem_qc_capture_active_) {
                const int32_t slot =
                    (static_cast<size_t>(il) < std_layer_slot_.size())
                        ? std_layer_slot_[il] : -1;
                if (slot >= 0) {
                    if (kvmem_query_end_ > kvmem_query_begin_) {
                        kvmem_capture_query_multi(
                            static_cast<uint32_t>(slot), chunk_off, batch,
                            base_pos, rope_base_pos, q_stride_buf);
                    }
                    // Build the full-coverage per-layer content index incrementally
                    // (#91): index EVERY block of this chunk from the freshly-RoPE'd
                    // K (de-RoPE'd at rope_base_pos == the bake position), not just
                    // the in-span question rows.
                    kvmem_capture_kbar_multi(static_cast<uint32_t>(slot), batch,
                                             base_pos, rope_base_pos, k_stride_buf);
                }
            }

            const uint32_t per_pos = standard_n_kv_heads * standard_head_dim;
            if (use_paged_prefill) {
                require_status(backend_.kv_append_batch_paged_device(
                    k_cache(il), *k_batch_, attn_base_pos, per_pos, batch,
                    attn_pages_dev, attn_pages_count, kv_page_size()));
                require_status(backend_.kv_append_batch_paged_device(
                    v_cache(il), *v_batch_, attn_base_pos, per_pos, batch,
                    attn_pages_dev, attn_pages_count, kv_page_size()));
                if (record_ops) record(report, "layer." + std::to_string(il) + ".kv_append_batch_paged");
            } else {
                require_status(backend_.kv_append_batch(
                    k_cache(il), *k_batch_, base_pos, per_pos, batch));
                require_status(backend_.kv_append_batch(
                    v_cache(il), *v_batch_, base_pos, per_pos, batch));
                if (record_ops) record(report, "layer." + std::to_string(il) + ".kv_append_batch");
            }

            const float scale = 1.0f / std::sqrt(static_cast<float>(standard_head_dim));
            if (use_paged_prefill) {
                if (batch == 1) {
                    require_status(backend_.attention_decode_batch_paged_gated_device(
                        *mid_batch_, *q_batch_, 2 * standard_head_dim,
                        attention_k_cache(il), v_cache(il),
                        attn_pages_dev, attn_pages_count, kv_page_size(),
                        standard_n_heads, standard_n_kv_heads,
                        standard_head_dim, attn_base_pos, batch,
                        q_stride_buf, mid_stride, scale));
                } else {
                    DeviceStatus attn_st = backend_.attention_prefill_batch_paged_gated_device(
                        *mid_batch_, *q_batch_, 2 * standard_head_dim,
                        attention_k_cache(il), v_cache(il),
                        attn_pages_dev, attn_pages_count, kv_page_size(),
                        standard_n_heads, standard_n_kv_heads,
                        standard_head_dim, attn_base_pos, batch,
                        q_stride_buf, mid_stride, scale);
                    if (!attn_st.ok) {
                        if (std::getenv("QW3_DEBUG_PREFILL_PLAN")) {
                            static int fb_dbg = 0;
                            if (fb_dbg < 4) {
                                std::fprintf(stderr,
                                    "[qw3] verify attn FELL BACK to batch-decode: %s\n",
                                    attn_st.message ? attn_st.message : "(null)");
                                ++fb_dbg;
                            }
                        }
                        require_status(backend_.attention_decode_batch_paged_gated(
                            *mid_batch_, *q_batch_, 2 * standard_head_dim,
                            attention_k_cache(il), v_cache(il),
                            chunk_bs ? window_pages_host_.data() : kv_page_indices(),
                            attn_pages_count, kv_page_size(),
                            standard_n_heads, standard_n_kv_heads,
                            standard_head_dim, attn_base_pos, batch,
                            q_stride_buf, mid_stride, scale));
                    }
                }
                if (record_ops) record(report, "layer." + std::to_string(il) + ".attention_sdpa_batch_paged");
            } else {
                require_status(backend_.attention_decode_batch_gated(
                    *mid_batch_, *q_batch_, 2 * standard_head_dim,
                    attention_k_cache(il), v_cache(il),
                    standard_n_heads, standard_n_kv_heads,
                    standard_head_dim, base_pos, batch,
                    q_stride_buf, mid_stride, scale));
                if (record_ops) record(report, "layer." + std::to_string(il) + ".attention_sdpa_batch");
            }

            require_status(backend_.q8_0_matmul(*attn_out_batch_, *layer.attn_output, *mid_batch_,
                                                 batch, mid_stride, h_stride));
        }

        require_status(backend_.add_n(*h_batch_, *h_batch_, *attn_out_batch_,
                                      static_cast<uint64_t>(batch) * h_stride));
        if (record_ops) record(report, "layer." + std::to_string(il) + ".attn_residual_batch");

        require_status(backend_.rms_norm_batch(*norm_batch_, *h_batch_, *layer.ffn_norm,
                                                batch, h_stride, eps));
        if (record_ops) record(report, "layer." + std::to_string(il) + ".ffn_norm_batch");

        require_status(backend_.q8_0_matmul(*ffn_gate_batch_, *layer.ffn_gate, *norm_batch_,
                                             batch, h_stride, ffn_stride));
        require_status(backend_.q8_0_matmul(*ffn_up_batch_, *layer.ffn_up, *norm_batch_,
                                             batch, h_stride, ffn_stride));
        // Batched scratch buffers are capacity-sized; only the active `batch`
        // rows hold valid data. silu_mul/add over the full o.count would
        // process batch_capacity_ rows (e.g. the 2048-wide prefill chunk) for
        // a 2..5-row verify batch — that elementwise overhead, not the matmul,
        // was the verify FFN's dominant cost. Cap to batch rows via *_n.
        require_status(backend_.silu_mul_n(*ffn_mid_batch_, *ffn_gate_batch_, *ffn_up_batch_,
                                           static_cast<uint64_t>(batch) * ffn_stride));
        require_status(backend_.q8_0_matmul(*ffn_out_batch_, *layer.ffn_down, *ffn_mid_batch_,
                                             batch, ffn_stride, h_stride));
        require_status(backend_.add_n(*h_batch_, *h_batch_, *ffn_out_batch_,
                                      static_cast<uint64_t>(batch) * h_stride));
        if (record_ops) record(report, "layer." + std::to_string(il) + ".ffn_batch");
        }

        if (kvmem_immutable_source_k_) {
            kvmem_flush_raw_k_capture(base_pos, 0, batch);
        }
        if (chunk_bs) window_query_pos_ += batch;
        if (kvmem_gpu_page_pool_ && !mtp_single_chunk) {
            kvmem_register_until(base_pos + batch);
            // Window-baked chunk: K/V were physically RoPE'd at rope_base_pos (the
            // WINDOW frame), not the true position, yet register_append recorded
            // baked_pos == orig_pos_start (true). Correct the bookkeeping to the
            // ACTUAL bake position so the reselect below — and every later
            // de-rotate (set_selection -> assemble, canonicalize-for-tier) —
            // cancels the real rotation. delta == 0 when not window-active or
            // under NO_REROPE, so this is a no-op below budget and byte-identical.
            if (chunk_bs &&
                (kvmem_fix_bakedpos_ || kvmem_immutable_source_k_) &&
                block_store_) {
                const int64_t delta = static_cast<int64_t>(rope_base_pos) -
                                      static_cast<int64_t>(base_pos);
                if (delta != 0) {
                    const auto &blks = block_store_->blocks();
                    for (uint32_t bi = static_cast<uint32_t>(blks.size());
                         bi-- > 0;) {
                        const KvMemBlock &b = blks[bi];
                        // Blocks are ordered by ascending orig_pos_start; stop at
                        // the first one fully before this chunk's true range.
                        if (static_cast<uint32_t>(b.orig_pos_start) + b.n_tokens <=
                            base_pos) {
                            break;
                        }
                        const int64_t desired_bake =
                            static_cast<int64_t>(b.orig_pos_start) + delta;
                        block_store_->set_block_baked_pos(
                            bi, desired_bake);
                    }
                }
            }
            if (!kvmem_mtp_tiered_) {
                // With no independently-built MTP cache, MAIN V is the final
                // spill source at this point. Start write-through before the
                // pressure check so D2H can overlap the next chunk.
                kvmem_prefill_writeback(base_pos + batch);
            }
            if (kvmem_defer_prefill_pressure_) {
                kvmem_deferred_prefill_tokens_ =
                    std::max(kvmem_deferred_prefill_tokens_, batch);
            } else {
                kvmem_maybe_prefill_offload(batch);
            }
        }

        if (try_capture) {
            require_status(backend_.end_capture());
            require_status(backend_.replay_graph());
        } else if (full_chunk) {
            // First full chunk: ran eager so backend-side scratch buffers
            // (q8_1 staging, fattn workspace) get sized before we attempt
            // capture on the next chunk.
            capture_warmup_pending = false;
        }
    }

    // Only the LAST prompt token's logits are needed to start decoding. The
    // last chunk leaves its rows in h_batch_ rows [0, last_chunk_batch).
    require_status(backend_.copy_d2d(*h_, *h_batch_, (last_chunk_batch - 1) * h_stride, h_stride));
    if (state_checkpoints && save_state_checkpoints > 0) {
        state_checkpoints->ready = true;
    }

    // MTP verify/replay: caller only wants KV + recurrent state advanced (and
    // optionally checkpoints), no logits at all.
    if (!compute_logits && !row_argmaxes) {
        position_ += total;
        require_status(backend_.end());
        report.ok = true;
        return report;
    }

    // MTP verify: per-row argmax over every token in the batch. Used by the
    // speculative verifier to compare drafted tokens against the target's
    // greedy continuation row-by-row.
    if (row_argmaxes) {
        const uint64_t vocab = weights_.output().rows;
        ensure_logits_batch_scratch(total);
        require_status(backend_.rms_norm_batch(*norm_batch_, *h_batch_, weights_.output_norm(),
                                                total, h_stride, eps));
        record(report, "output_norm_batch");
        require_status(backend_.q8_0_matmul(*logits_batch_, weights_.output(),
                                             *norm_batch_, total, h_stride,
                                             static_cast<uint32_t>(vocab)));
        record(report, "lm_head_batch");

        require_status(backend_.argmax_batch(*logits_batch_, total,
                                             static_cast<uint32_t>(vocab),
                                             *row_argmaxes));
        if (copy_last_logits) {
            require_status(backend_.copy_d2d(*logits_, *logits_batch_,
                                             static_cast<uint64_t>(total - 1) * vocab,
                                             vocab));
        }
        // MTP speculative-sampling accept test needs the full per-row target
        // distribution on host, not just the argmax. Copy each verify row from
        // the [total × vocab] device scratch before the next forward overwrites
        // it. Only done when sampling is active (temp>0), so greedy MTP pays
        // nothing here.
        if (row_logits_host) {
            row_logits_host->assign(total, std::vector<float>(vocab));
            for (uint32_t r = 0; r < total; ++r) {
                require_status(backend_.copy_to_host(
                    *logits_batch_, (*row_logits_host)[r].data(),
                    static_cast<uint64_t>(r) * vocab, vocab));
            }
        }
        require_status(backend_.end());

        position_ += total;
        const DeviceArgmax &best = row_argmaxes->back();
        report.argmax_token = best.token;
        report.argmax_logit = best.logit;
        report.argmax_text = model_.gguf().token_text(static_cast<uint32_t>(best.token));
        record(report, "lm_head_argmax_batch");
        report.ok = true;
        return report;
    }

    require_status(backend_.rms_norm(*norm_, *h_, weights_.output_norm(), eps));
    record(report, "output_norm");
    require_status(backend_.q8_0_matvec(*logits_, weights_.output(), *norm_));
    const DeviceArgmax best = backend_.argmax(*logits_);
    require_status(backend_.end());

    position_ += total;
    report.argmax_token = best.token;
    report.argmax_logit = best.logit;
    report.argmax_text = model_.gguf().token_text(static_cast<uint32_t>(best.token));
    record(report, "lm_head_argmax");
    report.ok = true;
    return report;
}

bool QwenExecutor::copy_last_logits(std::vector<float> &out) const {
    if (!logits_) return false;
    out.resize(static_cast<size_t>(logits_->count));
    const DeviceStatus st = backend_.copy_to_host(*logits_, out.data(), 0, logits_->count);
    return st.ok;
}

// ===========================================================================
// MTP (Multi-Token Prediction) draft head + speculative-decode plumbing.
// Ported from qw3_ly, adapted to qw3's backend op signatures (3-arg
// q8_0_matvec_add / 4-arg q8_0_matvec_silu_mul with eager fallback).
// ===========================================================================

NativeExecutorReport QwenExecutor::forward_mtp_draft(uint32_t token_id) {
    ensure_mtp_scratch();
    return forward_mtp_draft_from(token_id, *h_, position_, 0, 1);
}

std::vector<NativeExecutorReport> QwenExecutor::forward_mtp_draft_chain(uint32_t token_id,
                                                                        uint32_t max_tokens) {
    std::vector<NativeExecutorReport> reports;
    if (max_tokens == 0) return reports;
    ensure_mtp_scratch();
    uint32_t current = token_id;
    for (uint32_t i = 0; i < max_tokens; ++i) {
        if (i >= kv_ctx_size_) break;
        const DeviceTensor &h_input = (i == 0) ? *h_ : *mtp_h_;
        NativeExecutorReport report = forward_mtp_draft_from(current, h_input,
                                                             position_ + i, i, i + 1);
        const int next = report.argmax_token;
        reports.push_back(std::move(report));
        if (next < 0) break;
        current = static_cast<uint32_t>(next);
    }
    return reports;
}

std::vector<NativeExecutorReport> QwenExecutor::forward_mtp_draft_chain_with_prefix(uint32_t token_id,
                                                                                   uint32_t max_tokens) {
    std::vector<NativeExecutorReport> reports;
    if (max_tokens == 0) return reports;
    ensure_mtp_scratch();
    if (position_ > mtp_prefix_len_) {
        NativeExecutorReport report;
        report.missing_kernels.push_back("native MTP prefix KV is behind target position");
        reports.push_back(std::move(report));
        return reports;
    }
    // kvmem: run the speculative draft over the same re-RoPE'd window the verify
    // path uses (selected blocks only). Each step writes speculative K at window
    // slot window_query_pos_+i, aliasing the true MTP tail page just allocated;
    // attention scans the window page list, so draft cost is bounded by the
    // window and the draft agrees with the window-aware verify. The chain is
    // speculative: position_/window_query_pos_/mtp_prefix_len_ are NOT advanced,
    // and the aliased tail pages are trimmed off the window list afterwards
    // (commit-time priming re-primes the accepted token at its true position).
    // When kvmem is off this branch is bypassed and the draft is byte-identical
    // to the legacy true-frame path below.
    const bool window_mode = kvmem_active_;
    const uint32_t rope_limit = model_.config().n_ctx_train;
    if (window_mode && rope_limit > 0) {
        if (window_query_pos_ >= rope_limit) return reports;
        max_tokens = std::min<uint32_t>(
            max_tokens, rope_limit - window_query_pos_);
    }
    const uint32_t pre_window_pages = mtp_window_page_count_;
    uint32_t current = token_id;
    for (uint32_t i = 0; i < max_tokens; ++i) {
        const uint32_t cache_pos = position_ + i;
        if (cache_pos >= kv_ctx_size_) break;
        const DeviceTensor &h_input = (i == 0) ? *h_ : *mtp_h_;
        NativeExecutorReport report;
        if (window_mode) {
            mtp_kv_pages_.ensure_pages(backend_, kv_ctx_size_, cache_pos, 1);
            kvmem_extend_mtp_window_for_decode_n(i + 1, position_);
            const uint32_t win_pos = window_query_pos_ + i;
            report = forward_mtp_draft_from(current, h_input, win_pos, win_pos,
                                            win_pos + 1, /*compute_logits=*/true,
                                            /*argmax_out=*/nullptr,
                                            /*argmax_out_index=*/0,
                                            /*token_source=*/nullptr,
                                            /*token_source_index=*/0,
                                            /*window_frame=*/true);
        } else {
            report = forward_mtp_draft_from(current, h_input,
                                            cache_pos, cache_pos,
                                            cache_pos + 1);
        }
        const bool ok = report.ok;
        const int next = report.argmax_token;
        reports.push_back(std::move(report));
        if (!ok || next < 0) break;
        if (!window_mode && i == 0) {
            mtp_prefix_len_ = std::max<uint32_t>(mtp_prefix_len_, position_ + 1);
        }
        current = static_cast<uint32_t>(next);
    }
    if (window_mode && mtp_window_page_count_ != pre_window_pages) {
        mtp_window_pages_host_.resize(pre_window_pages);
        mtp_window_page_count_ = pre_window_pages;
    }
    return reports;
}

std::vector<NativeExecutorReport> QwenExecutor::forward_mtp_draft_chain_with_prefix_device(uint32_t token_id,
                                                                                           uint32_t max_tokens) {
    if (max_tokens == 0) return {};
    ensure_mtp_scratch();
    if (!mtp_draft_argmaxes_ || mtp_draft_argmax_capacity_ < max_tokens) {
        mtp_draft_argmaxes_ = backend_.argmax_buffer(max_tokens);
        mtp_draft_argmax_capacity_ = mtp_draft_argmaxes_ ? max_tokens : 0;
    }
    if (!mtp_draft_argmaxes_) {
        return forward_mtp_draft_chain_with_prefix(token_id, max_tokens);
    }
    if (position_ > mtp_prefix_len_) {
        NativeExecutorReport report;
        report.missing_kernels.push_back("native MTP prefix KV is behind target position");
        return {std::move(report)};
    }

    // kvmem windowed draft (see forward_mtp_draft_chain_with_prefix): drive the
    // window frame here too while preserving on-device argmax chaining (the
    // argmax buffer / token_source are independent of the KV frame).
    const bool window_mode = kvmem_active_;
    const uint32_t rope_limit = model_.config().n_ctx_train;
    if (window_mode && rope_limit > 0) {
        if (window_query_pos_ >= rope_limit) return {};
        max_tokens = std::min<uint32_t>(
            max_tokens, rope_limit - window_query_pos_);
    }
    const uint32_t pre_window_pages = mtp_window_page_count_;
    std::vector<NativeExecutorReport> reports;
    reports.reserve(max_tokens);
    for (uint32_t i = 0; i < max_tokens; ++i) {
        const uint32_t cache_pos = position_ + i;
        if (cache_pos >= kv_ctx_size_) break;
        const DeviceTensor &h_input = (i == 0) ? *h_ : *mtp_h_;
        const DeviceArgmaxBuffer *token_source = i == 0 ? nullptr : mtp_draft_argmaxes_.get();
        uint32_t draft_pos = cache_pos;
        bool window_frame = false;
        if (window_mode) {
            mtp_kv_pages_.ensure_pages(backend_, kv_ctx_size_, cache_pos, 1);
            kvmem_extend_mtp_window_for_decode_n(i + 1, position_);
            draft_pos = window_query_pos_ + i;
            window_frame = true;
        }
        NativeExecutorReport report = forward_mtp_draft_from(token_id,
                                                             h_input,
                                                             draft_pos,
                                                             draft_pos,
                                                             draft_pos + 1,
                                                             /*compute_logits=*/true,
                                                             mtp_draft_argmaxes_.get(),
                                                             i,
                                                             token_source,
                                                             i == 0 ? 0 : i - 1,
                                                             window_frame);
        const bool ok = report.ok;
        reports.push_back(std::move(report));
        if (!ok) break;
        if (!window_mode && i == 0) {
            mtp_prefix_len_ = std::max<uint32_t>(mtp_prefix_len_, position_ + 1);
        }
    }
    if (window_mode && mtp_window_page_count_ != pre_window_pages) {
        mtp_window_pages_host_.resize(pre_window_pages);
        mtp_window_page_count_ = pre_window_pages;
    }

    if (reports.empty() || !reports.back().ok) return reports;
    std::vector<DeviceArgmax> host(reports.size());
    if (auto st = backend_.copy_argmax_buffer_to_host(*mtp_draft_argmaxes_,
                                                      host.data(),
                                                      static_cast<uint32_t>(host.size()));
        !st.ok) {
        NativeExecutorReport report;
        report.missing_kernels.push_back(st.message);
        return {std::move(report)};
    }
    for (size_t i = 0; i < reports.size(); ++i) {
        reports[i].argmax_token = host[i].token;
        reports[i].argmax_logit = host[i].logit;
    }
    return reports;
}

NativeExecutorReport QwenExecutor::forward_mtp_draft_from(uint32_t token_id,
                                                          const DeviceTensor &h_input,
                                                          uint32_t rope_pos,
                                                          uint32_t cache_pos,
                                                          uint32_t seq_len,
                                                          bool compute_logits,
                                                          DeviceArgmaxBuffer *argmax_out,
                                                          uint32_t argmax_out_index,
                                                          const DeviceArgmaxBuffer *token_source,
                                                          uint32_t token_source_index,
                                                          bool window_frame,
                                                          bool kv_only) {
    NativeExecutorReport report;
    const NativePlanInfo &plan = model_.plan();
    if (!plan.mtp_supported) {
        report.missing_kernels.push_back("native MTP plan is incomplete");
        return report;
    }
    const QwenMtpWeights *mtp = weights_.mtp();
    if (!mtp || !mtp->eh_proj || !mtp->embed_tokens || !mtp->enorm ||
        !mtp->hnorm || !mtp->shared_head_head || !mtp->shared_head_norm) {
        report.missing_kernels.push_back("native MTP weights are incomplete");
        return report;
    }
    if (mtp->layer.recurrent) {
        report.missing_kernels.push_back("native MTP recurrent draft layer is not supported");
        return report;
    }
    ensure_mtp_scratch();
    if (seq_len == 0 || cache_pos >= kv_ctx_size_ || seq_len > kv_ctx_size_) {
        report.missing_kernels.push_back("native MTP KV cache is too small for requested draft chain");
        return report;
    }

    const QwenConfig &cfg = model_.config();
    if (cfg.n_ctx_train > 0 && rope_pos >= cfg.n_ctx_train) {
        report.missing_kernels.push_back(
            "native MTP RoPE position exceeds the model context limit");
        return report;
    }
    const uint32_t standard_head_dim = cfg.head_dim;
    const uint32_t standard_n_heads = cfg.n_heads;
    const uint32_t standard_n_kv_heads = cfg.n_kv_heads;
    const float eps = cfg.rms_eps;
    const QwenLayerWeights &layer = mtp->layer;

    require_status(backend_.begin());
    begin_record_timing(executor_trace_timing_enabled());

    if (token_source) {
        require_status(backend_.q8_0_get_row_from_argmax(*mtp_embd_,
                                                         *mtp->embed_tokens,
                                                         *token_source,
                                                         token_source_index));
    } else {
        require_status(backend_.q8_0_get_row(*mtp_embd_, *mtp->embed_tokens, token_id));
    }
    record(report, "mtp.token_embedding_lookup");
    require_status(backend_.rms_norm(*mtp_enorm_, *mtp_embd_, *mtp->enorm, eps));
    record(report, "mtp.enorm");
    require_status(backend_.rms_norm(*mtp_hnorm_, h_input, *mtp->hnorm, eps));
    record(report, "mtp.hnorm");
    require_status(backend_.copy_d2d_into(*mtp_concat_, 0, *mtp_enorm_, 0, cfg.n_embd));
    require_status(backend_.copy_d2d_into(*mtp_concat_, cfg.n_embd, *mtp_hnorm_, 0, cfg.n_embd));
    record(report, "mtp.concat");
    require_status(backend_.q8_0_matvec(*mtp_h_, *mtp->eh_proj, *mtp_concat_));
    record(report, "mtp.eh_proj");

    require_status(backend_.rms_norm(*norm_, *mtp_h_, *layer.attn_norm, eps));
    record(report, "mtp.attn_norm");
    {
        DeviceTensor *outs[3] = {q_.get(), k_.get(), v_.get()};
        const DeviceWeight *ws[3] = {layer.attn_q, layer.attn_k, layer.attn_v};
        require_status(backend_.q8_0_matvec_fanout(outs, ws, 3, *norm_));
    }
    record(report, "mtp.attention_qkv_projection");
    require_status(backend_.rmsnorm_per_head(*q_, *layer.attn_q_norm,
                                             standard_n_heads,
                                             2 * standard_head_dim,
                                             standard_head_dim, eps));
    require_status(backend_.rmsnorm_per_head(*k_, *layer.attn_k_norm,
                                             standard_n_kv_heads,
                                             standard_head_dim,
                                             standard_head_dim, eps));
    // Prefix priming is the one point where the MTP head's projected K exists
    // before RoPE.  Keep that position-free value as the immutable authority;
    // selected-window K is reconstructed from it at the current compact slot.
    // Draft-chain rows (window_frame=true, kv_only=false) are speculative and
    // must never overwrite the accepted-prefix authority.
    if (kvmem_mtp_local_positions_ && kv_only && !window_frame) {
        kvmem_capture_raw_mtp_k(*k_, cache_pos, 1);
    }
    trace_rope_position_if_out_of_range(
        "forward_mtp_draft_from.qk", rope_pos, 1, cfg.n_ctx_train,
        static_cast<int32_t>(weights_.n_layers()), /*kernel_uses=*/2);
    require_status(backend_.rope_partial(*q_, standard_n_heads,
                                         2 * standard_head_dim,
                                         cfg.rope_dim, rope_pos, cfg.rope_theta));
    require_status(backend_.rope_partial(*k_, standard_n_kv_heads,
                                         standard_head_dim,
                                         cfg.rope_dim, rope_pos, cfg.rope_theta));

    const uint32_t per_pos = standard_n_kv_heads * standard_head_dim;
    // kvmem windowed draft (window_frame=true, only when kvmem_active_): attend
    // the same re-RoPE'd window the verify path uses. The window pages alias the
    // already-primed physical MTP pages, so no ensure_pages/validate is needed;
    // the page table is the lockstep MTP window built in kvmem_assemble.
    if (!window_frame) {
        mtp_kv_pages_.ensure_pages(backend_, kv_ctx_size_, cache_pos, 1);
        if (external_mtp_kv_cache_) {
            mtp_kv_pages_.validate_physical_capacity(
                external_mtp_kv_cache_->physical_slots, "external MTP");
        }
    }
    // Under MTP tiering the cache is a bounded paged pool (pool_pages*page_size
    // slots), NOT a dense full-context buffer: a dense append by absolute
    // position would land out of the pool's range. Force the paged append so
    // prime writes (window_frame=false) route through mtp_kv_pages_ too.
    const bool use_paged_prefix =
        mtp_paged_prefix_enabled() || window_frame || kvmem_mtp_tiered_;
    const DeviceTensor &mtp_pages_dev =
        window_frame ? *mtp_window_pages_device_ : mtp_kv_pages_.device_indices();
    const uint32_t mtp_pages_count =
        window_frame ? mtp_window_page_count_ : mtp_kv_pages_.count();
    const uint32_t mtp_page_size = mtp_kv_pages_.page_size;
    DeviceTensor &mtp_k = mtp_k_cache();
    DeviceTensor &mtp_v = mtp_v_cache();
    if (use_paged_prefix) {
        require_status(backend_.kv_append_batch_paged_device(
            mtp_k, *k_, cache_pos, per_pos,
            1, mtp_pages_dev,
            mtp_pages_count, mtp_page_size));
        require_status(backend_.kv_append_batch_paged_device(
            mtp_v, *v_, cache_pos, per_pos,
            1, mtp_pages_dev,
            mtp_pages_count, mtp_page_size));
    } else {
        require_status(backend_.kv_append(mtp_k, *k_, cache_pos, per_pos));
        require_status(backend_.kv_append(mtp_v, *v_, cache_pos, per_pos));
    }
    record(report, "mtp.kv_append");

    // Prefix-prime fast path: when this forward only exists to populate the MTP
    // KV cache at the true token position (commit-time prefix maintenance), the
    // self-attention + residual + FFN below produce a hidden state that is never
    // read — the prime callers take mtp_prefix_h_ from the MAIN hidden, not from
    // this MTP forward, and the next draft chain reads only the appended K/V. So
    // we stop right after kv_append, turning the prime from O(ctx) attention into
    // O(1) per token. Persistent state (mtp KV, mtp_prefix_h_, mtp_prefix_len_)
    // is bit-identical to running the full forward, so acceptance and output are
    // unchanged for both plain-MTP and kvmem-MTP.
    if (kv_only) {
        require_status(backend_.end());
        report.ok = true;
        return report;
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(standard_head_dim));
    if (use_paged_prefix) {
        require_status(backend_.attention_decode_batch_paged_gated_device(
            *mid_, *q_, 2 * standard_head_dim, mtp_k,
            mtp_v, mtp_pages_dev,
            mtp_pages_count, mtp_page_size,
            standard_n_heads, standard_n_kv_heads, standard_head_dim,
            seq_len - 1, 1, 2 * standard_head_dim,
            standard_n_heads * standard_head_dim, scale));
    } else {
        require_status(backend_.attention_decode(*mid_, *scores_, *q_,
                                                 2 * standard_head_dim,
                                                 mtp_k, mtp_v,
                                                 standard_n_heads, standard_n_kv_heads,
                                                 standard_head_dim,
                                                 seq_len, scale));
        require_status(backend_.apply_attn_gate(*mid_, *q_,
                                                2 * standard_head_dim,
                                                standard_n_heads,
                                                standard_head_dim));
    }
    record(report, "mtp.attention_sdpa");
    // h += W_out * mid (fused matvec+add with eager fallback, mirroring the
    // main decode path).
    if (auto st = backend_.q8_0_matvec_add(*mtp_h_, *layer.attn_output, *mid_); !st.ok) {
        require_status(backend_.q8_0_matvec(*attn_out_, *layer.attn_output, *mid_));
        require_status(backend_.add(*mtp_h_, *mtp_h_, *attn_out_));
    }
    record(report, "mtp.attn_residual");

    require_status(backend_.rms_norm(*norm_, *mtp_h_, *layer.ffn_norm, eps));
    record(report, "mtp.ffn_norm");
    if (auto st = backend_.q8_0_matvec_silu_mul(*ffn_mid_, *layer.ffn_gate,
                                                *layer.ffn_up, *norm_);
        !st.ok) {
        DeviceTensor *outs[2] = {ffn_gate_.get(), ffn_up_.get()};
        const DeviceWeight *ws[2] = {layer.ffn_gate, layer.ffn_up};
        require_status(backend_.q8_0_matvec_fanout(outs, ws, 2, *norm_));
        require_status(backend_.silu_mul(*ffn_mid_, *ffn_gate_, *ffn_up_));
    }
    if (auto st = backend_.q8_0_matvec_add(*mtp_h_, *layer.ffn_down, *ffn_mid_); !st.ok) {
        require_status(backend_.q8_0_matvec(*ffn_out_, *layer.ffn_down, *ffn_mid_));
        require_status(backend_.add(*mtp_h_, *mtp_h_, *ffn_out_));
    }
    record(report, "mtp.ffn");

    if (!compute_logits) {
        require_status(backend_.end());
        report.ok = true;
        return report;
    }

    require_status(backend_.rms_norm(*norm_, *mtp_h_, *mtp->shared_head_norm, eps));
    record(report, "mtp.shared_head_norm");
    require_status(backend_.q8_0_matvec(*logits_, *mtp->shared_head_head, *norm_));
    DeviceArgmax best;
    if (argmax_out) {
        require_status(backend_.argmax_to_buffer(*logits_, *argmax_out, argmax_out_index));
    } else {
        best = backend_.argmax(*logits_);
    }
    require_status(backend_.end());

    if (!argmax_out) {
        report.argmax_token = best.token;
        report.argmax_logit = best.logit;
        report.argmax_text = model_.gguf().token_text(static_cast<uint32_t>(best.token));
    }
    record(report, "mtp.lm_head_argmax");
    report.ok = true;
    return report;
}

void QwenExecutor::commit_mtp_prefix(uint32_t prefix_len) {
    mtp_prefix_len_ = std::max<uint32_t>(mtp_prefix_len_,
                                         std::min<uint32_t>(prefix_len, kv_ctx_size_));
}

void QwenExecutor::commit_mtp_prefix_from_current_hidden(uint32_t prefix_len) {
    ensure_mtp_scratch();
    if (h_ && mtp_prefix_h_) {
        require_status(backend_.copy_d2d(*mtp_prefix_h_, *h_, 0, h_->count));
    }
    commit_mtp_prefix(prefix_len);
}

QwenExecutor::StateSnapshot QwenExecutor::snapshot_state() {
    StateSnapshot snapshot;
    capture_state(snapshot);
    return snapshot;
}

void QwenExecutor::capture_state(StateSnapshot &snapshot) {
    ensure_scratch();
    snapshot.position = position_;
    snapshot.kv_logical_pages = kv_pages_.count();
    snapshot.mtp_prefix_len = mtp_prefix_len_;
    snapshot.kvmem_registered_pos = kvmem_registered_pos_;
    // kvmem window state (inert unless kvmem is active for this session).
    snapshot.kvmem_active = kvmem_active_;
    snapshot.window_query_pos = window_query_pos_;
    snapshot.window_page_count = window_page_count_;
    if (h_) {
        if (!snapshot.h || snapshot.h->count != h_->count) {
            snapshot.h = backend_.scratch_f32(h_->count, "snapshot_h");
        }
        require_status(backend_.copy_d2d(*snapshot.h, *h_, 0, h_->count));
    }
    if (snapshot.recurrent_states.size() != recurrent_states_.size()) {
        snapshot.recurrent_states.resize(recurrent_states_.size());
    }
    if (snapshot.conv_states.size() != conv_states_.size()) {
        snapshot.conv_states.resize(conv_states_.size());
    }
    for (size_t i = 0; i < recurrent_states_.size(); ++i) {
        if (recurrent_states_[i]) {
            if (!snapshot.recurrent_states[i] ||
                snapshot.recurrent_states[i]->count != recurrent_states_[i]->count) {
                snapshot.recurrent_states[i] =
                    backend_.scratch_f32(recurrent_states_[i]->count, "snapshot_recurrent");
            }
            require_status(backend_.copy_d2d(*snapshot.recurrent_states[i],
                                             *recurrent_states_[i],
                                             0, recurrent_states_[i]->count));
        }
        if (conv_states_[i]) {
            if (!snapshot.conv_states[i] ||
                snapshot.conv_states[i]->count != conv_states_[i]->count) {
                snapshot.conv_states[i] =
                    backend_.scratch_f32(conv_states_[i]->count, "snapshot_conv");
            }
            require_status(backend_.copy_d2d(*snapshot.conv_states[i],
                                             *conv_states_[i],
                                             0, conv_states_[i]->count));
        }
    }
    snapshot.ready = true;
}

void QwenExecutor::restore_state(const StateSnapshot &snapshot) {
    if (!snapshot.ready) {
        throw std::runtime_error("cannot restore an empty QwenExecutor snapshot");
    }
    ensure_scratch();
    if (snapshot.h && h_) {
        require_status(backend_.copy_d2d(*h_, *snapshot.h, 0, h_->count));
    }
    for (size_t i = 0; i < recurrent_states_.size(); ++i) {
        if (recurrent_states_[i] && i < snapshot.recurrent_states.size() &&
            snapshot.recurrent_states[i]) {
            require_status(backend_.copy_d2d(*recurrent_states_[i],
                                             *snapshot.recurrent_states[i],
                                             0, recurrent_states_[i]->count));
        }
        if (conv_states_[i] && i < snapshot.conv_states.size() &&
            snapshot.conv_states[i]) {
            require_status(backend_.copy_d2d(*conv_states_[i],
                                             *snapshot.conv_states[i],
                                             0, conv_states_[i]->count));
        }
    }
    position_ = snapshot.position;
    kvmem_registered_pos_ = snapshot.kvmem_registered_pos;
    kv_pages_.truncate_to_logical_pages(snapshot.kv_logical_pages);
    mtp_prefix_len_ = std::min<uint32_t>(mtp_prefix_len_,
                                         snapshot.mtp_prefix_len);
    // Roll the kvmem window back to where it was at capture. Verify/decode only
    // ever appends pages at the window tail (kvmem_extend_window_for_decode), so
    // truncating the host page list + count and restoring window_query_pos_ is
    // sufficient: the surviving window slots keep their original re-RoPE bake.
    // No device re-upload is needed — the appended tail pages are simply no
    // longer addressed (window_page_count_ caps what attention/append read).
    if (snapshot.kvmem_active) {
        window_query_pos_ = snapshot.window_query_pos;
        window_page_count_ = snapshot.window_page_count;
        if (window_pages_host_.size() > window_page_count_) {
            window_pages_host_.resize(window_page_count_);
        }
        // MTP draft window is lockstep with the main window (same blocks); roll
        // its tail back so the next draft chain sees the pre-verify window. Clamp
        // the count to the actual host length -- the MTP window only grows during
        // the draft (not the verify that just ran), so its host can legitimately
        // be shorter than window_page_count_; the next draft's extend regrows it.
        // Keeping count == host size is the invariant the extend relies on.
        mtp_window_page_count_ = std::min<uint32_t>(
            window_page_count_,
            static_cast<uint32_t>(mtp_window_pages_host_.size()));
        if (mtp_window_pages_host_.size() > mtp_window_page_count_) {
            mtp_window_pages_host_.resize(mtp_window_page_count_);
        }
    }
}

void QwenExecutor::restore_recurrent_state(const StateSnapshot &snapshot) {
    if (!snapshot.ready) {
        throw std::runtime_error(
            "cannot restore recurrent state from an empty snapshot");
    }
    ensure_scratch();
    if (snapshot.recurrent_states.size() != recurrent_states_.size() ||
        snapshot.conv_states.size() != conv_states_.size()) {
        throw std::runtime_error(
            "recurrent-state snapshot layer count mismatch");
    }
    for (size_t i = 0; i < recurrent_states_.size(); ++i) {
        if (recurrent_states_[i]) {
            if (!snapshot.recurrent_states[i] ||
                snapshot.recurrent_states[i]->count !=
                    recurrent_states_[i]->count) {
                throw std::runtime_error(
                    "recurrent-state snapshot tensor mismatch at layer " +
                    std::to_string(i));
            }
            require_status(backend_.copy_d2d(
                *recurrent_states_[i], *snapshot.recurrent_states[i], 0,
                recurrent_states_[i]->count));
        }
        if (conv_states_[i]) {
            if (!snapshot.conv_states[i] ||
                snapshot.conv_states[i]->count != conv_states_[i]->count) {
                throw std::runtime_error(
                    "conv-state snapshot tensor mismatch at layer " +
                    std::to_string(i));
            }
            require_status(backend_.copy_d2d(
                *conv_states_[i], *snapshot.conv_states[i], 0,
                conv_states_[i]->count));
        }
    }
}

void QwenExecutor::kvmem_begin_query_replay(
        const StateSnapshot &boundary,
        const std::vector<uint32_t> &context_block_ids,
        bool reset_recurrent_state) {
    if (!boundary.ready) {
        throw std::runtime_error(
            "KVMem query replay requires a ready boundary snapshot");
    }
    if (!kvmem_enabled_ || !block_store_) {
        throw std::runtime_error(
            "KVMem query replay requires an enabled block store");
    }
    if (kvmem_pending_reselect_) {
        throw std::runtime_error(
            "KVMem query replay cannot start during a pending reselect");
    }
    const uint32_t bt = std::max<uint32_t>(
        1, block_store_->config().block_tokens);
    const uint32_t page_size = std::max<uint32_t>(1, kv_pages_.page_size);
    if (boundary.position > position_ || boundary.position % bt != 0 ||
        boundary.position % page_size != 0) {
        throw std::runtime_error(
            "KVMem query replay boundary must be an earlier block/page-aligned "
            "position");
    }
    for (uint32_t id : context_block_ids) {
        if (id >= block_store_->block_count() ||
            block_store_->blocks()[id].orig_pos_start >= boundary.position) {
            throw std::runtime_error(
                "KVMem query replay context selection crosses replay boundary");
        }
    }

    // First remove the selected query/suffix blocks while the complete store and
    // its tier metadata are still intact. This leaves the exact semantic context
    // selection assembled in the window and frees precisely the pool capacity
    // that the replayed suffix will consume.
    (void)kvmem_set_selection(context_block_ids);
    // The suffix block IDs are about to be recycled. Finish bounded async SSD
    // ownership first so an old-generation write cannot land after truncation.
    kvmem_finish_proactive_d2h(/*wait_all=*/true);
    kvmem_reap_pending_writes(/*wait_all=*/true);

    // Restore only the model-compute state at the boundary. restore_state() is
    // intentionally not used: it also restores the OLD pressure-window length,
    // whereas query replay must retain the final semantic context window above.
    ensure_scratch();
    if (boundary.h && h_) {
        require_status(backend_.copy_d2d(*h_, *boundary.h, 0, h_->count));
    }
    for (size_t i = 0; i < recurrent_states_.size(); ++i) {
        if (recurrent_states_[i] && i < boundary.recurrent_states.size() &&
            boundary.recurrent_states[i]) {
            if (reset_recurrent_state) {
                require_status(backend_.zero_tensor(*recurrent_states_[i]));
            } else {
                require_status(backend_.copy_d2d(
                    *recurrent_states_[i], *boundary.recurrent_states[i], 0,
                    recurrent_states_[i]->count));
            }
        }
        if (conv_states_[i] && i < boundary.conv_states.size() &&
            boundary.conv_states[i]) {
            if (reset_recurrent_state) {
                require_status(backend_.zero_tensor(*conv_states_[i]));
            } else {
                require_status(backend_.copy_d2d(
                    *conv_states_[i], *boundary.conv_states[i], 0,
                    conv_states_[i]->count));
            }
        }
    }
    position_ = boundary.position;
    kvmem_registered_pos_ = boundary.position;
    mtp_prefix_len_ = std::min<uint32_t>(mtp_prefix_len_,
                                         boundary.mtp_prefix_len);

    // The replay boundary is block/page aligned, so the complete suffix can be
    // discarded without sharing a physical page with preserved context.
    const uint32_t logical_pages = boundary.position / page_size;
    kv_pages_.truncate_to_logical_pages(logical_pages);
    if (mtp_kv_pages_.page_size > 0) {
        const uint32_t mtp_pages =
            boundary.position / mtp_kv_pages_.page_size;
        mtp_kv_pages_.truncate_to_logical_pages(mtp_pages);
    }

    // Reconcile the block store and release any CPU/NVMe slots belonging to the
    // removed suffix. Prefix mean-K slices remain position-invariant and valid;
    // the replayed suffix is invalidated logically and overwritten as its final
    // K rows are produced below.
    std::vector<KvMemDroppedBlock> dropped =
        block_store_->truncate_to(boundary.position);
    kvmem_writeback_next_block_ = std::min<uint32_t>(
        kvmem_writeback_next_block_, block_store_->block_count());
    for (const KvMemDroppedBlock &d : dropped) {
        if (d.cpu_slot >= 0 && kvmem_cpu_tier_) {
            kvmem_cpu_tier_->release_block(d.block_id);
            kvmem_release_cpu_slot(d.cpu_slot);
        }
        if (d.nvme_slot >= 0 && kvmem_nvme_tier_) {
            kvmem_nvme_tier_->release_block(d.block_id);
        }
    }
    if (boundary.position < kvmem_raw_k_valid_tokens_.size()) {
        std::fill(
            kvmem_raw_k_valid_tokens_.begin() + boundary.position,
            kvmem_raw_k_valid_tokens_.end(), static_cast<uint8_t>(0));
    }
    if (boundary.position < kvmem_raw_mtp_k_valid_tokens_.size()) {
        std::fill(
            kvmem_raw_mtp_k_valid_tokens_.begin() + boundary.position,
            kvmem_raw_mtp_k_valid_tokens_.end(), static_cast<uint8_t>(0));
    }
    kvmem_truncate_raw_k(boundary.position);
    if (mtp_baked_pos_.size() > block_store_->block_count()) {
        mtp_baked_pos_.resize(block_store_->block_count());
    }

    g_content_ready_ = false;
    g_indexed_blocks_ = 0;
    g_kbar_multi_ready_ = false;
    g_kbar_multi_blocks_ = 0;
    kvmem_qc_captured_tokens_ = boundary.position;
    kvmem_qc_captured_blocks_ = boundary.position / bt;

    // Capture a fresh query representation and rebuild the replayed content-index
    // suffix from the same K rows that are permanently retained.
    kvmem_drain_query_capture();
    g_query_multi_count_ = 0;
    g_query_multi_ready_ = false;
    g_query_ready_ = false;
    kvmem_dn_qcount_ = 0;
    kvmem_dn_ready_ = false;
    kvmem_query_replay_active_ = true;

    if (std::getenv("QW3_KVMEM_TRACE")) {
        std::fprintf(stderr,
                     "[bs-query-replay] begin boundary=%u context_blocks=%zu "
                     "window_tokens=%u reset_recurrent=%d\n",
                     boundary.position, context_block_ids.size(),
                     window_query_pos_, reset_recurrent_state ? 1 : 0);
    }
}

void QwenExecutor::kvmem_end_query_replay() {
    if (!kvmem_query_replay_active_) return;
    kvmem_register_until(position_);
    kvmem_query_replay_active_ = false;
    if (std::getenv("QW3_KVMEM_TRACE")) {
        std::fprintf(stderr,
                     "[bs-query-replay] end position=%u window_tokens=%u "
                     "query_rows=%u ready=%d\n",
                     position_, window_query_pos_, g_query_multi_count_,
                     g_query_multi_ready_ ? 1 : 0);
    }
}

void QwenExecutor::kvmem_start_query_prefetch(
        uint32_t replay_suffix_tokens) {
    auto trace_skip = [&](const char *reason) {
        if (kvmem_perf_trace_flag() || std::getenv("QW3_KVMEM_TRACE")) {
            std::fprintf(
                stderr,
                "[kvmem-query-prefetch] phase=skip reason=%s "
                "enabled=%d prefetch_active=%d pending_reselect=%d "
                "suffix_tokens=%u\n",
                reason, kvmem_query_prefetch_enabled_ ? 1 : 0,
                kvmem_prefetch_.active ? 1 : 0,
                kvmem_pending_reselect_ ? 1 : 0,
                replay_suffix_tokens);
        }
    };
    if (!kvmem_query_prefetch_enabled_ || !block_store_ ||
        !kvmem_gpu_page_pool_ || !kvmem_cpu_tier_ ||
        kvmem_prefetch_.active || kvmem_pending_reselect_ ||
        replay_suffix_tokens == 0) {
        trace_skip("prerequisite");
        return;
    }

    const KvMemStoreConfig &cfg = block_store_->config();
    const uint32_t page_size = std::max<uint32_t>(1, kv_pages_.page_size);
    const uint32_t pages_per_block =
        std::max<uint32_t>(1, cfg.block_tokens / page_size);
    uint32_t free_pages = kvmem_gpu_page_pool_->free_pages();
    if (kvmem_mtp_gpu_page_pool_) {
        free_pages = std::min(
            free_pages, kvmem_mtp_gpu_page_pool_->free_pages());
    }
    // The suffix is appended after prefetch starts. Keep all of its pages plus
    // one full block as allocator slack; speculative pages may use only the
    // remaining generation reserve.
    const uint32_t suffix_pages =
        (replay_suffix_tokens + page_size - 1) / page_size;
    const uint64_t reserve_pages =
        static_cast<uint64_t>(suffix_pages) + pages_per_block;
    if (free_pages <= reserve_pages) {
        trace_skip("no_generation_reserve_pages");
        return;
    }
    uint32_t candidate_page_budget =
        free_pages - static_cast<uint32_t>(reserve_pages);
    const uint32_t max_blocks = std::max<uint32_t>(
        1, env_uint32_or("QW3_KVMEM_QUERY_PREFETCH_BLOCKS", 512));

    struct Candidate {
        uint32_t block_id = 0;
        double score = 0.0;
    };
    std::vector<Candidate> ranked;
    const auto &blocks = block_store_->blocks();
    ranked.reserve(blocks.size());
    for (const KvMemBlock &b : blocks) {
        if (b.tier != KvTier::CPU || b.cpu_slot < 0 ||
            kvmem_block_pages_resident(b)) {
            continue;
        }
        // profile_score is query-independent and therefore available before
        // the first-pass query has produced its semantic mean-K score. The
        // recency tie-break makes the choice deterministic when no heat has
        // yet been accumulated.
        ranked.push_back(Candidate{b.block_id, b.profile_score});
    }
    std::sort(
        ranked.begin(), ranked.end(),
        [](const Candidate &a, const Candidate &b) {
            if (a.score != b.score) return a.score > b.score;
            return a.block_id > b.block_id;
        });

    KvMemPlan advisory;
    advisory.remaps.reserve(
        std::min<size_t>(ranked.size(), max_blocks));
    for (const Candidate &candidate : ranked) {
        if (advisory.remaps.size() >= max_blocks) break;
        const KvMemBlock &block = blocks[candidate.block_id];
        const uint32_t first_page = block.orig_pos_start / page_size;
        const uint32_t last_page =
            (block.orig_pos_start + block.n_tokens - 1) / page_size;
        const uint32_t block_pages = last_page - first_page + 1;
        if (block_pages > candidate_page_budget) continue;
        candidate_page_budget -= block_pages;
        KvMemRemap rm;
        rm.block_id = candidate.block_id;
        rm.n_tokens = block.n_tokens;
        advisory.remaps.push_back(rm);
    }
    if (advisory.remaps.empty()) {
        trace_skip(ranked.empty()
                       ? "no_cpu_candidates"
                       : "candidate_page_budget");
        return;
    }

    kvmem_query_prefetch_blocks_.clear();
    kvmem_query_prefetch_blocks_.reserve(advisory.remaps.size());
    for (const KvMemRemap &rm : advisory.remaps) {
        kvmem_query_prefetch_blocks_.push_back(rm.block_id);
    }
    kvmem_query_prefetch_start_ns_ = kvmem_steady_ns();
    kvmem_query_prefetch_active_ = true;
    try {
        kvmem_start_prefetch(advisory);
        // Ordinary semantic stage-in defers CPU gathering so a separate
        // assembly worker can start first. Here the overlap partner is the
        // upcoming query prefill, so enqueue the bounded first two slabs now.
        if (kvmem_prefetch_.bulk_cpu &&
            !kvmem_prefetch_.cpu_reads.empty()) {
            kvmem_submit_prefetch_cpu_batches();
        }
    } catch (...) {
        kvmem_query_prefetch_active_ = false;
        kvmem_query_prefetch_blocks_.clear();
        kvmem_query_prefetch_start_ns_ = 0;
        throw;
    }
    if (kvmem_perf_trace_flag() || std::getenv("QW3_KVMEM_TRACE")) {
        std::fprintf(
            stderr,
            "[kvmem-query-prefetch] phase=start candidates=%zu "
            "suffix_tokens=%u free_pages=%u reserved_pages=%llu\n",
            kvmem_query_prefetch_blocks_.size(), replay_suffix_tokens,
            free_pages,
            static_cast<unsigned long long>(reserve_pages));
    }
}

// ARCHIVED (2026-07-23): the DeltaNet recurrent-state artifact implementation
// was used only by the frozen LongMemEval-M rebuilt-state diagnostic. The public
// request entry is disabled in qw3_server.cpp; compile this implementation out
// as well so normal inference cannot perform state-file I/O or state replacement.
#if 0
namespace {

constexpr uint64_t kRebuiltStateMagic = 0x3154535244335751ULL; // "QW3DRST1"
constexpr uint32_t kRebuiltStateVersion = 1;

template <typename T>
void rebuilt_state_write(std::ofstream &out, const T &value,
                         const std::string &path) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(T));
    if (!out) {
        throw std::runtime_error(
            "failed to write KVMem rebuilt recurrent state: " + path);
    }
}

template <typename T>
void rebuilt_state_read(std::ifstream &in, T &value,
                        const std::string &path) {
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error(
            "truncated KVMem rebuilt recurrent state: " + path);
    }
}

} // namespace

void QwenExecutor::kvmem_export_recurrent_state(
    const std::string &path,
    const std::vector<uint32_t> &source_tokens) const {
    if (path.empty()) {
        throw std::runtime_error(
            "KVMem rebuilt recurrent-state export path is empty");
    }
    const std::string tmp = path + ".tmp";
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error(
            "cannot create KVMem rebuilt recurrent state: " + tmp);
    }
    rebuilt_state_write(out, kRebuiltStateMagic, tmp);
    rebuilt_state_write(out, kRebuiltStateVersion, tmp);
    const uint32_t layer_count =
        static_cast<uint32_t>(recurrent_states_.size());
    rebuilt_state_write(out, layer_count, tmp);
    const uint64_t token_count = source_tokens.size();
    rebuilt_state_write(out, token_count, tmp);
    if (!source_tokens.empty()) {
        out.write(reinterpret_cast<const char *>(source_tokens.data()),
                  static_cast<std::streamsize>(source_tokens.size() *
                                               sizeof(uint32_t)));
        if (!out) {
            throw std::runtime_error(
                "failed to write rebuilt-state source tokens: " + tmp);
        }
    }

    for (size_t il = 0; il < recurrent_states_.size(); ++il) {
        const uint64_t recurrent_count =
            recurrent_states_[il] ? recurrent_states_[il]->count : 0;
        const uint64_t conv_count =
            conv_states_[il] ? conv_states_[il]->count : 0;
        rebuilt_state_write(out, recurrent_count, tmp);
        rebuilt_state_write(out, conv_count, tmp);
        if (recurrent_count > 0) {
            std::vector<float> host(recurrent_count);
            require_status(backend_.copy_to_host(
                *recurrent_states_[il], host.data(), 0, recurrent_count));
            out.write(reinterpret_cast<const char *>(host.data()),
                      static_cast<std::streamsize>(host.size() * sizeof(float)));
            if (!out) {
                throw std::runtime_error(
                    "failed to write rebuilt recurrent tensor: " + tmp);
            }
        }
        if (conv_count > 0) {
            std::vector<float> host(conv_count);
            require_status(backend_.copy_to_host(
                *conv_states_[il], host.data(), 0, conv_count));
            out.write(reinterpret_cast<const char *>(host.data()),
                      static_cast<std::streamsize>(host.size() * sizeof(float)));
            if (!out) {
                throw std::runtime_error(
                    "failed to write rebuilt conv tensor: " + tmp);
            }
        }
    }
    out.flush();
    if (!out) {
        throw std::runtime_error(
            "failed to flush KVMem rebuilt recurrent state: " + tmp);
    }
    out.close();
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        throw std::runtime_error(
            "failed to publish KVMem rebuilt recurrent state: " + path);
    }
    if (std::getenv("QW3_KVMEM_TRACE")) {
        std::fprintf(stderr,
                     "[bs-rebuilt-state] export path=%s tokens=%zu layers=%u\n",
                     path.c_str(), source_tokens.size(), layer_count);
    }
}

void QwenExecutor::kvmem_import_recurrent_state(
    const std::string &path,
    const std::vector<uint32_t> &expected_source_tokens) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(
            "cannot open KVMem rebuilt recurrent state: " + path);
    }
    uint64_t magic = 0;
    uint32_t version = 0;
    uint32_t layer_count = 0;
    uint64_t token_count = 0;
    rebuilt_state_read(in, magic, path);
    rebuilt_state_read(in, version, path);
    rebuilt_state_read(in, layer_count, path);
    rebuilt_state_read(in, token_count, path);
    if (magic != kRebuiltStateMagic || version != kRebuiltStateVersion) {
        throw std::runtime_error(
            "unsupported KVMem rebuilt recurrent-state format: " + path);
    }
    if (layer_count != recurrent_states_.size()) {
        throw std::runtime_error(
            "KVMem rebuilt recurrent-state layer count mismatch: " + path);
    }
    if (token_count != expected_source_tokens.size()) {
        throw std::runtime_error(
            "KVMem rebuilt recurrent-state source token count mismatch: " +
            path);
    }
    std::vector<uint32_t> source_tokens(static_cast<size_t>(token_count));
    if (!source_tokens.empty()) {
        in.read(reinterpret_cast<char *>(source_tokens.data()),
                static_cast<std::streamsize>(source_tokens.size() *
                                             sizeof(uint32_t)));
        if (!in) {
            throw std::runtime_error(
                "truncated rebuilt-state source token vector: " + path);
        }
    }
    if (source_tokens != expected_source_tokens) {
        const auto mismatch = std::mismatch(
            source_tokens.begin(), source_tokens.end(),
            expected_source_tokens.begin(), expected_source_tokens.end());
        const size_t index =
            static_cast<size_t>(mismatch.first - source_tokens.begin());
        throw std::runtime_error(
            "KVMem rebuilt recurrent-state source token mismatch at index " +
            std::to_string(index) + ": " + path);
    }

    // Validate every tensor shape and read the complete file into host buffers
    // before mutating device state. A bad or mismatched artifact can therefore
    // never leave a partially imported recurrent state behind.
    struct LayerState {
        std::vector<float> recurrent;
        std::vector<float> conv;
    };
    std::vector<LayerState> layers(layer_count);
    for (size_t il = 0; il < layer_count; ++il) {
        uint64_t recurrent_count = 0;
        uint64_t conv_count = 0;
        rebuilt_state_read(in, recurrent_count, path);
        rebuilt_state_read(in, conv_count, path);
        const uint64_t expected_recurrent =
            recurrent_states_[il] ? recurrent_states_[il]->count : 0;
        const uint64_t expected_conv =
            conv_states_[il] ? conv_states_[il]->count : 0;
        if (recurrent_count != expected_recurrent ||
            conv_count != expected_conv) {
            throw std::runtime_error(
                "KVMem rebuilt recurrent-state tensor shape mismatch at layer " +
                std::to_string(il) + ": " + path);
        }
        layers[il].recurrent.resize(static_cast<size_t>(recurrent_count));
        layers[il].conv.resize(static_cast<size_t>(conv_count));
        if (recurrent_count > 0) {
            in.read(reinterpret_cast<char *>(layers[il].recurrent.data()),
                    static_cast<std::streamsize>(recurrent_count *
                                                 sizeof(float)));
            if (!in) {
                throw std::runtime_error(
                    "truncated rebuilt recurrent tensor at layer " +
                    std::to_string(il) + ": " + path);
            }
        }
        if (conv_count > 0) {
            in.read(reinterpret_cast<char *>(layers[il].conv.data()),
                    static_cast<std::streamsize>(conv_count * sizeof(float)));
            if (!in) {
                throw std::runtime_error(
                    "truncated rebuilt conv tensor at layer " +
                    std::to_string(il) + ": " + path);
            }
        }
    }
    char trailing = 0;
    if (in.read(&trailing, 1)) {
        throw std::runtime_error(
            "KVMem rebuilt recurrent-state file has trailing bytes: " + path);
    }

    for (size_t il = 0; il < layer_count; ++il) {
        if (recurrent_states_[il]) {
            require_status(backend_.copy_bytes_from_host(
                *recurrent_states_[il], 0, layers[il].recurrent.data(),
                layers[il].recurrent.size() * sizeof(float)));
        }
        if (conv_states_[il]) {
            require_status(backend_.copy_bytes_from_host(
                *conv_states_[il], 0, layers[il].conv.data(),
                layers[il].conv.size() * sizeof(float)));
        }
    }
    if (std::getenv("QW3_KVMEM_TRACE")) {
        std::fprintf(stderr,
                     "[bs-rebuilt-state] import path=%s tokens=%zu layers=%u\n",
                     path.c_str(), expected_source_tokens.size(), layer_count);
    }
}
#endif

void QwenExecutor::kvmem_reset_recurrent_state() {
    for (size_t i = 0; i < recurrent_states_.size(); ++i) {
        if (recurrent_states_[i]) {
            require_status(backend_.zero_tensor(*recurrent_states_[i]));
        }
        if (conv_states_[i]) {
            require_status(backend_.zero_tensor(*conv_states_[i]));
        }
    }
    if (std::getenv("QW3_KVMEM_TRACE")) {
        std::fprintf(stderr,
                     "[bs-query-replay] reset recurrent/conv state at "
                     "position=%u\n",
                     position_);
    }
}

void QwenExecutor::seed_from_shared_prefix(const std::vector<int32_t> &shared_pages,
                                           const StateSnapshot &recur,
                                           uint32_t aligned_len) {
    ensure_scratch();
    const uint32_t page_size = kv_pages_.page_size;
    if (page_size == 0) {
        throw std::runtime_error("seed_from_shared_prefix: zero KV page size");
    }
    if (static_cast<uint64_t>(shared_pages.size()) * page_size != aligned_len) {
        throw std::runtime_error(
            "seed_from_shared_prefix: aligned_len must be a whole number of "
            "KV pages");
    }
    // The page table must be empty (caller resets the executor first).
    kv_pages_.adopt_shared_pages(backend_, shared_pages);
    // Restore recurrent + conv state captured at exactly aligned_len. This
    // also sets position_ and truncates kv logical pages to recur.kv_logical_pages.
    // Because we adopted exactly the shared pages, that truncation is a no-op.
    restore_state(recur);
    position_ = aligned_len;
}

std::vector<int32_t> QwenExecutor::mark_kv_prefix_shared(uint32_t logical_start_page) {
    // Mark the prefix [0..logical_start_page) as borrowed so this executor's
    // dtor/reset won't free those pages — the prefix cache now pins them. The
    // suffix [logical_start_page..end) stays owned and frees normally when the
    // live request finishes. Returns the prefix's physical pages for the cache
    // to record + pin.
    const uint32_t n = std::min<uint32_t>(
        logical_start_page, static_cast<uint32_t>(kv_pages_.pages.size()));
    if (kv_pages_.owned.size() < kv_pages_.pages.size()) {
        kv_pages_.owned.resize(kv_pages_.pages.size(), true);
    }
    for (uint32_t i = 0; i < n; ++i) kv_pages_.owned[i] = false;
    return std::vector<int32_t>(kv_pages_.pages.begin(),
                                kv_pages_.pages.begin() + n);
}

std::vector<int32_t> QwenExecutor::kv_physical_pages() const {
    return kv_pages_.pages;
}

// ---- Block-sparse KV attention ------------------------------------------

void QwenExecutor::configure_kvmem(const KvMemStoreConfig &cfg) {
    kvmem_finish_proactive_d2h(/*wait_all=*/true);
    kvmem_reap_pending_writes(/*wait_all=*/true);
    kvmem_free_write_slabs_.clear();
    kvmem_free_writeback_slabs_.clear();
    kvmem_proactive_d2h_batches_.clear();
    kvmem_free_stageout_slabs_.clear();
    kvmem_free_read_slabs_.clear();
    kvmem_prefill_writeback_enabled_ = false;
    kvmem_query_prefetch_enabled_ = false;
    kvmem_query_prefetch_active_ = false;
    kvmem_query_prefetch_blocks_.clear();
    kvmem_query_prefetch_start_ns_ = 0;
    kvmem_inclusive_cpu_backing_ = false;
    kvmem_mtp_incremental_assembly_ = false;
    kvmem_writeback_slab_count_ = 0;
    kvmem_read_slab_count_ = 0;
    kvmem_writeback_next_block_ = 0;
    kvmem_writeback_d2h_bytes_ = 0;
    kvmem_writeback_d2h_wait_ns_ = 0;
    kvmem_writeback_d2h_batches_ = 0;
    kvmem_writeback_blocks_ = 0;
    // v1 requires block_tokens to be a positive multiple of the KV page size so
    // every block boundary is page-aligned: the window packs selected blocks
    // contiguously, and a non-aligned block would split a physical page across
    // two logical window slots (corrupting the byte-offset math). The default
    // block_tokens=128 / page_size=16 satisfies this.
    const uint32_t page_size = kv_pages_.page_size;
    if (cfg.block_tokens == 0 || page_size == 0 ||
        (cfg.block_tokens % page_size) != 0) {
        throw std::runtime_error(
            "block-sparse requires --kvmem-block-tokens to be a positive multiple "
            "of the KV page size (" + std::to_string(page_size) + ")");
    }
    KvMemStoreConfig effective = cfg;
    if (effective.optimization_level > KvMemOptimizationLevel::Opt3) {
        throw std::runtime_error(
            "--kvmem-optimization-level requests an optimization level that "
            "is reserved but not implemented in this build; use kvmem_init or "
            "opt_1, opt_2, or opt_3");
    }
    if (!effective.legacy_optimization_profile) {
        static constexpr const char *kLegacyOptimizationEnvs[] = {
            "QW3_KVMEM_PREFILL_WRITEBACK",
            "QW3_KVMEM_ASSEMBLY_MODE",
            "QW3_KVMEM_RAW_K_BLOCK_MAJOR",
            "QW3_KVMEM_OVERLAP_STAGEIN_ASSEMBLY",
            "QW3_KVMEM_PERSISTENT_CPU_POOL",
            "QW3_KVMEM_INCLUSIVE_CPU",
            "QW3_KVMEM_QUERY_PREFETCH",
            "QW3_KVMEM_MTP_INCREMENTAL_ASSEMBLY",
            "QW3_KVMEM_RAW_K_TRANSFER_BLOCKS",
            "QW3_KVMEM_PREFILL_CPU_ADMIT",
        };
        for (const char *name : kLegacyOptimizationEnvs) {
            if (std::getenv(name)) {
                throw std::runtime_error(
                    std::string(name) +
                    " is a legacy optimization switch and is not accepted "
                    "in the default all-on mode; use "
                    "--kvmem-optimize-off instead");
            }
        }
    }
    // Immutable source K is the default. The legacy environment override is
    // still tri-state: an explicit false value disables it just like the CLI
    // opt-out, while an unset variable preserves EngineOptions.
    kvmem_immutable_source_k_ = env_flag_enabled(
        "QW3_KVMEM_IMMUTABLE_SOURCE_K", effective.immutable_source_k);
    effective.immutable_source_k = kvmem_immutable_source_k_;
    kvmem_rope_sincos_.reset();
    kvmem_rope_table_positions_ = 0;
    kvmem_rope_table_explicit_ = false;
    kvmem_raw_pipeline_enabled_ = false;
    kvmem_raw_transfer_block_cap_ = 128;
    for (auto &slot : kvmem_raw_pipeline_slots_) {
        slot = RawMaterializeSlot{};
    }
    kvmem_rope_table_enabled_ =
        kvmem_immutable_source_k_ &&
        std::strcmp(backend_.name(), "cuda-device") == 0;
    // Opt3 is the end-to-end transfer pipeline profile.  Once immutable raw K
    // is available, make the corresponding two-slot gather/H2D/scatter
    // assembly pipeline part of that profile as well.  Lower profiles keep
    // the bitwise-equivalent table path without the extra staging slot.
    kvmem_raw_pipeline_enabled_ =
        kvmem_rope_table_enabled_ &&
        effective.packed_rematerialization;
    if (effective.legacy_optimization_profile) {
        if (const char *mode = std::getenv("QW3_KVMEM_ASSEMBLY_MODE")) {
            kvmem_rope_table_explicit_ = true;
            if (std::strcmp(mode, "legacy") == 0) {
                kvmem_rope_table_enabled_ = false;
                kvmem_raw_pipeline_enabled_ = false;
            } else if (std::strcmp(mode, "table") == 0 ||
                       std::strcmp(mode, "pipeline") == 0) {
                if (!kvmem_immutable_source_k_) {
                    throw std::runtime_error(
                        "KVMem table assembly currently requires immutable "
                        "source K so out-of-table baked positions can be rebuilt");
                }
                kvmem_rope_table_enabled_ = true;
                kvmem_raw_pipeline_enabled_ =
                    std::strcmp(mode, "pipeline") == 0;
            } else {
                throw std::runtime_error(
                    "QW3_KVMEM_ASSEMBLY_MODE must be legacy, table, or pipeline");
            }
        }
    }
    if (effective.optimization_level >= KvMemOptimizationLevel::Opt2 &&
        effective.nvme_tier_bytes > 0 &&
        !kvmem_immutable_source_k_) {
        throw std::runtime_error(
            "KVMem opt_2 inclusive clean backing requires immutable raw K; "
            "remove --no-kvmem-immutable-k");
    }
    const char *kvmem_kv_dtype = std::getenv("QW3_KV_DTYPE");
    if (!kvmem_immutable_source_k_ && kvmem_kv_dtype &&
        std::strcmp(kvmem_kv_dtype, "fp8") == 0) {
        throw std::runtime_error(
            "KVMem fp8 requires --kvmem-immutable-k so lossy re-RoPE does "
            "not accumulate across reselections");
    }
    if (kvmem_immutable_source_k_) {
        if (kvmem_kv_dtype && std::strcmp(kvmem_kv_dtype, "q8") == 0) {
            throw std::runtime_error(
                "--kvmem-immutable-k currently requires fp16, "
                "fp32, or fp8 K; q8 row-scale re-RoPE is unsupported");
        }
    }
    if (!effective.legacy_optimization_profile &&
        !kvmem_immutable_source_k_) {
        if (effective.proactive_stage_out) {
            throw std::runtime_error(
                "KVMem proactive-stage-out requires immutable raw K; "
                "remove --no-kvmem-immutable-k or add "
                "--kvmem-optimize-off proactive-stage-out");
        }
        if (effective.packed_rematerialization) {
            throw std::runtime_error(
                "KVMem packed-rematerialization requires immutable raw K; "
                "remove --no-kvmem-immutable-k or add "
                "--kvmem-optimize-off packed-rematerialization");
        }
        if (!effective.hierarchical_reuse) {
            throw std::runtime_error(
                "KVMem hierarchical-reuse-off requires immutable raw K "
                "to force a position-safe full reload");
        }
    }
    if (effective.gpu_memory_ratio < 0.0) effective.gpu_memory_ratio = 0.0;
    if (effective.gpu_memory_ratio > 1.0) effective.gpu_memory_ratio = 1.0;
    if (effective.gpu_low_watermark < 0.0) effective.gpu_low_watermark = 0.0;
    if (effective.gpu_low_watermark > 1.0) effective.gpu_low_watermark = 1.0;
    if (effective.gpu_high_watermark < effective.gpu_low_watermark) {
        effective.gpu_high_watermark = effective.gpu_low_watermark;
    }
    if (effective.gpu_high_watermark > 1.0) effective.gpu_high_watermark = 1.0;

    const QwenConfig &model_cfg = model_.config();
    const uint32_t standard_layers =
        count_standard_attention_layers(model_cfg, weights_.n_layers());
    if (kvmem_immutable_source_k_) {
        effective.immutable_max_baked_position = model_cfg.n_ctx_train;
        // The end-to-end accuracy control found a deterministic FP16 answer
        // regression at 32 resident moves. Use the conservative limit of 8 for
        // both FP16 and FP8. Cold blocks still rebuild unconditionally, and an
        // environment override remains available for controlled experiments.
        if (!effective.legacy_optimization_profile) {
            effective.immutable_refresh_remaps = 8u;
        }
        if (const char *env =
                std::getenv("QW3_KVMEM_IMMUTABLE_REFRESH_REMAPS")) {
            const long long v = std::atoll(env);
            if (v >= 0) {
                effective.immutable_refresh_remaps =
                    static_cast<uint32_t>(v);
            }
        }
        if (const char *env =
                std::getenv("QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS")) {
            const long long v = std::atoll(env);
            if (v >= 0) {
                effective.immutable_refresh_abs_delta_tokens =
                    static_cast<uint64_t>(v);
            }
        }
        if (const char *env =
                std::getenv("QW3_KVMEM_IMMUTABLE_MAX_BAKED_POSITION")) {
            const long long v = std::atoll(env);
            if (v >= 0) {
                effective.immutable_max_baked_position =
                    static_cast<uint64_t>(v);
            }
        }
    }
    if (kvmem_rope_table_enabled_) {
        const uint64_t max_position = std::min<uint64_t>(
            kv_ctx_size_,
            std::max<uint64_t>(
                effective.select_budget,
                effective.immutable_max_baked_position));
        if (max_position == 0 ||
            max_position > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error(
                "KVMem RoPE table position range is invalid");
        }
        kvmem_rope_table_positions_ =
            static_cast<uint32_t>(max_position);
    }
    // If MTP tiering will engage (predicate mirrored exactly in the pool-engage
    // block below), the MTP layer rides the same per-block blobs as a trailing
    // segment, so the CPU/NVMe slot_bytes must budget for one extra layer. This
    // MUST agree with kvmem_block_spill_bytes' layer count or the copy-in guard
    // (src_bytes < expected) trips.
    const bool mtp_will_tier =
        effective.mtp_enabled && weights_.mtp() != nullptr &&
        external_mtp_kv_cache_ == nullptr &&
        kvmem_mtp_tier_enabled();
    kvmem_mtp_local_positions_ =
        kvmem_immutable_source_k_ && mtp_will_tier &&
        kvmem_mtp_local_positions_enabled();
    // Re-applying a RoPE delta to the mutable MTP working K is faster than
    // rebuilding from the position-free authority, but it is not numerically
    // equivalent: FP8/FP16 rounding accumulates across reselections.  A
    // deterministic temp=0 AgentLongBench A/B changed the greedy completion
    // when this path was enabled, while rebuilding every selected MTP K from
    // raw authority reproduced kvmem_init byte-for-byte.  Keep the incremental
    // path available only as an explicit experimental opt-in; optimization
    // levels must not silently change model output.
    kvmem_mtp_incremental_assembly_ =
        kvmem_mtp_local_positions_ &&
        effective.legacy_optimization_profile &&
        effective.optimization_level >= KvMemOptimizationLevel::Opt3 &&
        env_flag_enabled("QW3_KVMEM_MTP_INCREMENTAL_ASSEMBLY", false);
    if (kvmem_mtp_local_positions_ && model_cfg.n_ctx_train > 0 &&
        static_cast<uint64_t>(effective.select_budget) +
                effective.gen_budget >
            model_cfg.n_ctx_train) {
        throw std::runtime_error(
            "KVMem local-position MTP requires --kvmem-budget + "
            "--kvmem-gen-budget to fit inside the model context limit (" +
            std::to_string(effective.select_budget) + " + " +
            std::to_string(effective.gen_budget) + " > " +
            std::to_string(model_cfg.n_ctx_train) + ")");
    }
    const uint64_t standard_kv_block_bytes =
        estimate_kvmem_block_bytes(model_cfg, standard_layers,
                                   effective.block_tokens);
    const uint64_t mtp_kv_block_bytes = mtp_will_tier
        ? estimate_kvmem_block_bytes(model_cfg, 1, effective.block_tokens)
        : 0;
    // The raw-K CPU mirror owns standard-layer K in immutable mode, so ordinary
    // CPU/NVMe tier records contain standard-layer V only. Local-position MTP
    // likewise stores raw K separately and spills only V; the legacy MTP path
    // retains K+V records.
    const uint64_t mtp_spill_block_bytes =
        kvmem_mtp_local_positions_
            ? mtp_kv_block_bytes / 2u  // immutable MTP raw-K lives separately
            : mtp_kv_block_bytes;
    effective.estimated_block_bytes = kvmem_immutable_source_k_
        ? standard_kv_block_bytes / 2u + mtp_spill_block_bytes
        : standard_kv_block_bytes + mtp_spill_block_bytes;
    const uint64_t gpu_resident_block_bytes =
        standard_kv_block_bytes + mtp_kv_block_bytes;

    uint64_t raw_k_mirror_max_bytes = 0;
    uint64_t raw_k_row_bytes = 0;
    uint64_t raw_mtp_k_mirror_max_bytes = 0;
    uint64_t capacity_cpu_tier_bytes = effective.cpu_tier_bytes;
    if (kvmem_immutable_source_k_) {
        if (const char *env =
                std::getenv("QW3_KVMEM_RAW_K_CHUNK_TOKENS")) {
            const long long v = std::atoll(env);
            if (v > 0) {
                kvmem_raw_k_chunk_tokens_ = static_cast<uint32_t>(
                    std::min<long long>(
                        v, std::numeric_limits<uint32_t>::max()));
            }
        }
        kvmem_raw_k_chunk_tokens_ = std::max<uint32_t>(
            effective.block_tokens, kvmem_raw_k_chunk_tokens_);
        kvmem_raw_k_chunk_tokens_ =
            ((kvmem_raw_k_chunk_tokens_ + effective.block_tokens - 1) /
             effective.block_tokens) * effective.block_tokens;
        uint64_t elem_bytes = 2;
        if (kvmem_kv_dtype && std::strcmp(kvmem_kv_dtype, "fp32") == 0) {
            elem_bytes = 4;
        } else if (kvmem_kv_dtype &&
                   std::strcmp(kvmem_kv_dtype, "fp8") == 0) {
            elem_bytes = 1;
        }
        raw_k_row_bytes =
            static_cast<uint64_t>(model_cfg.n_kv_heads) *
            model_cfg.head_dim * elem_bytes;
        const uint64_t raw_chunk_count =
            (static_cast<uint64_t>(kv_ctx_size_) +
             kvmem_raw_k_chunk_tokens_ - 1) /
            kvmem_raw_k_chunk_tokens_;
        const uint64_t padded_raw_tokens =
            raw_chunk_count * kvmem_raw_k_chunk_tokens_;
        raw_k_mirror_max_bytes =
            static_cast<uint64_t>(standard_layers) * padded_raw_tokens *
            raw_k_row_bytes;
        if (kvmem_mtp_local_positions_) {
            raw_mtp_k_mirror_max_bytes =
                padded_raw_tokens * raw_k_row_bytes;
        }
        const uint64_t raw_authority_bytes =
            raw_k_mirror_max_bytes + raw_mtp_k_mirror_max_bytes;
        if (effective.cpu_tier_bytes < raw_authority_bytes) {
            throw std::runtime_error(
                "KVMem immutable raw-K chunks can grow to " +
                std::to_string(raw_authority_bytes) +
                " CPU bytes, exceeding --kvmem-cpu-gb (" +
                std::to_string(effective.cpu_tier_bytes) +
                " bytes). Raise --kvmem-cpu-gb, lower --ctx, or use fp8 KV.");
        }
        // Runtime raw-K and CPU V blocks share the full host budget dynamically.
        // Capacity validation still reserves the worst-case raw-K authority:
        // when a prompt reaches --ctx, only this remainder is guaranteed to be
        // available to the CPU spill cache and all other V blocks need NVMe.
        capacity_cpu_tier_bytes =
            effective.cpu_tier_bytes - raw_authority_bytes;
    }
    const uint64_t total_device_bytes = backend_.total_device_bytes();
    if (effective.gpu_memory_ratio > 0.0 &&
        total_device_bytes > 0 &&
        gpu_resident_block_bytes > 0) {
        // The ratio bounds TOTAL process GPU usage (model weights + GPU-resident
        // KV + prefill scratch), not just the KV pool in isolation. Weights are
        // already resident at configure time, so the room left for the resident
        // KV pool is (ratio*total - already_used - scratch_reserve). Sizing the
        // pool from the raw ratio*total instead lets the whole KV stay GPU-
        // resident at intermediate contexts (e.g. 256K: 13.4 GiB KV < 47.8 GiB
        // raw cap), so the bounded pool never engages and never frees GPU pages
        // after stage-out -> process peak blows past the ratio ceiling. Reserving
        // weights + scratch forces the pool to engage and cap resident KV so the
        // ceiling actually holds. The adaptive prefill-chunk cap below keeps
        // scratch within the reserve at the longest contexts.
        const uint64_t ceiling = static_cast<uint64_t>(
            static_cast<long double>(total_device_bytes) *
            effective.gpu_memory_ratio);
        const uint64_t free_now = backend_.free_device_bytes();
        const uint64_t used_now =
            (free_now > 0 && free_now <= total_device_bytes)
                ? (total_device_bytes - free_now)
                : 0;
        // Reserve headroom under the ceiling for transient prefill scratch (FA2
        // attention workspace + split-K partials + q8_1 matmul staging +
        // per-token activation scratch = per_token_scratch_bytes()*chunk). Under
        // kvmem this scratch is NOT ctx-bound: attention is windowed to
        // select_budget and prefill runs in fixed chunks, so the transient peak
        // is bounded by the window/chunk, not the total context. Measured on
        // this model (budget 32768, chunk 2048): peak scratch over idle stays
        // flat at <0.8 GiB across 30K..150K-token prefills. A 3 GiB reserve is
        // ~4x that measured peak; the runtime chunk-floor (see the forward-time
        // free_device_bytes cap) plus the cudaMalloc hard-fail auto-chunk remain
        // the actual OOM backstop, so this static value only needs to be
        // realistic, not worst-case. The old 15 GiB flat reserve was calibrated
        // for the non-kvmem 2M-ctx dense path and over-subtracted ~14 GiB here,
        // spuriously tripping the budget+gen_reserve hard-error at ratio 0.5.
        // Tunable via QW3_KVMEM_SCRATCH_RESERVE_MIB.
        uint64_t scratch_reserve = static_cast<uint64_t>(3) << 30;
        if (const char *env = std::getenv("QW3_KVMEM_SCRATCH_RESERVE_MIB")) {
            const long long v = std::atoll(env);
            if (v >= 0) scratch_reserve = static_cast<uint64_t>(v) << 20;
        }
        const uint64_t reserved = used_now + scratch_reserve;
        const uint64_t cap_bytes =
            (ceiling > reserved + gpu_resident_block_bytes)
                ? (ceiling - reserved)
                : gpu_resident_block_bytes;
        const uint64_t cap_blocks64 = cap_bytes / gpu_resident_block_bytes;
        uint32_t cap_blocks = cap_blocks64 >
                static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
            ? std::numeric_limits<uint32_t>::max()
            : static_cast<uint32_t>(cap_blocks64);
        // cudaMemGetInfo is GLOBAL, so `used_now` (total - free) also counts any
        // co-tenant process on the GPU, and at very long ctx the weights alone
        // can approach the ceiling. Either drives cap_bytes to ~one block, which
        // (via the gpu_blocks>=2 gate below) DISABLES the bounded pool and falls
        // back to the DENSE full-ctx KV cache -- O(ctx) per standard layer, tens
        // of GiB, the exact OOM this pool exists to prevent. The bounded pool is
        // the memory-SAVING path, so whenever tiers can absorb the spill and the
        // context does not fit densely, floor the capacity to a minimum resident
        // window: enough blocks to hold the selection budget (kept above the
        // low-watermark shrink threshold below) plus a recent cushion and prefill
        // -chunk headroom. This window is small (~budget tokens of KV) and always
        // required for kvmem to function, so flooring never over-commits memory.
        const bool tiers_available =
            effective.estimated_block_bytes > 0 &&
            (effective.cpu_tier_bytes > 0 || effective.nvme_tier_bytes > 0);
        if (tiers_available) {
            const uint32_t budget_blocks_min = std::max<uint32_t>(
                1, effective.select_budget / effective.block_tokens);
            const uint32_t recent_blocks_min = effective.recent_blocks;
            const long double lw = effective.gpu_low_watermark > 0.0
                ? static_cast<long double>(effective.gpu_low_watermark)
                : 1.0L;
            const uint32_t budget_keep =
                static_cast<uint32_t>(
                    static_cast<long double>(budget_blocks_min) / lw) + 2u;
            const uint32_t min_window_blocks =
                budget_keep + recent_blocks_min + 16u;
            const uint64_t ctx_blocks_est =
                (static_cast<uint64_t>(kv_ctx_size_) + effective.block_tokens - 1) /
                effective.block_tokens;
            if (ctx_blocks_est > min_window_blocks && cap_blocks < min_window_blocks) {
                cap_blocks = min_window_blocks;
            }
        }
        effective.estimated_gpu_block_capacity = cap_blocks;
        if (cap_blocks > 0) {
            // The bounded GPU pool must hold BOTH the selection window
            // (select_budget) AND one turn's newly generated tokens
            // (gen_budget). In step update-mode no decode-time stage-out fires,
            // so a turn's register_append blocks stay GPU-resident; sizing the
            // pool to budget + gen_reserve makes mid-decode pool exhaustion
            // structurally impossible (paired with the server's max_tokens <=
            // gen_budget clamp). Hard-fail at configure time when VRAM can't fit
            // both, naming the knobs, instead of silently shrinking the window.
            const uint32_t sel_budget_blocks = std::max<uint32_t>(
                1, (effective.select_budget + effective.block_tokens - 1) /
                       effective.block_tokens);
            const uint32_t gen_reserve_blocks =
                (effective.gen_budget + effective.block_tokens - 1) /
                effective.block_tokens;
            const uint64_t required_blocks =
                static_cast<uint64_t>(sel_budget_blocks) + gen_reserve_blocks;
            if (required_blocks > cap_blocks) {
                throw std::runtime_error(
                    "kvmem: GPU pool cannot fit selection budget + generation "
                    "reserve. Need " + std::to_string(required_blocks) +
                    " blocks (" + std::to_string(sel_budget_blocks) +
                    " for --kvmem-budget + " + std::to_string(gen_reserve_blocks) +
                    " for --kvmem-gen-budget) but only " +
                    std::to_string(cap_blocks) + " fit under the current VRAM "
                    "ceiling. Raise --kvmem-gpu-memory-ratio, or lower "
                    "--kvmem-budget / --kvmem-gen-budget.");
            }
            uint64_t pool_blocks = required_blocks;
            // Full-context-query probe (task #50 follow-up). Size the GPU pool to
            // hold as much of the context as VRAM allows (up to the full ctx),
            // DECOUPLED from select_budget. With the whole prompt resident the
            // pressure-gated prefill offload (kvmem_maybe_prefill_offload) never
            // fires, so kvmem_active_ stays false through prefill and the trailing
            // question tokens attend over the ENTIRE context -> the captured
            // retrieval query is full-context, not recency-windowed. Selection and
            // decode still use select_budget: the post-prefill reselect assembles a
            // select_budget window from that full-context query. Env-gated; default
            // off -> pool == required_blocks -> byte-identical to today.
            if (std::getenv("QW3_KVMEM_FULLCTX_QUERY")) {
                const uint64_t ctx_blocks =
                    (static_cast<uint64_t>(kv_ctx_size_) +
                     effective.block_tokens - 1) / effective.block_tokens;
                pool_blocks = std::min<uint64_t>(cap_blocks, ctx_blocks);
                if (pool_blocks < required_blocks) pool_blocks = required_blocks;
            }
            effective.estimated_gpu_block_capacity =
                static_cast<uint32_t>(pool_blocks);
        }
    }

    block_store_ = std::make_unique<KvMemStore>(effective);
    kvmem_active_ = false;
    window_pages_host_.clear();
    window_page_count_ = 0;
    mtp_window_pages_host_.clear();
    mtp_window_page_count_ = 0;
    window_query_pos_ = 0;
    kvmem_cpu_tier_.reset();
    kvmem_nvme_tier_.reset();
    if (host_tier_pool_ && kvmem_cpu_bytes_) {
        host_tier_pool_->release(std::move(kvmem_cpu_bytes_));
    }
    kvmem_cpu_bytes_.reset();
    kvmem_stage_pinned_.reset();
    kvmem_prefetch_ = KvMemPrefetchState{};
    kvmem_query_prefetch_active_ = false;
    kvmem_query_prefetch_blocks_.clear();
    kvmem_query_prefetch_start_ns_ = 0;
    kvmem_pending_reselect_ = false;
    kvmem_pending_plan_ = KvMemPlan{};
    kvmem_registered_pos_ = 0;
    if (external_kv_cache_ == &kvmem_kv_cache_view_) {
        external_kv_cache_ = nullptr;
        kv_pages_.set_allocator(nullptr);
    }
    kvmem_gpu_page_pool_.reset();
    kvmem_k_cache_storage_.clear();
    kvmem_v_cache_storage_.clear();
    kvmem_kv_cache_view_ = KvCacheStorage{};
    kvmem_raw_k_chunks_.clear();
    kvmem_raw_k_mirror_bytes_ = 0;
    kvmem_raw_k_row_bytes_ = 0;
    kvmem_raw_k_valid_tokens_.clear();
    kvmem_raw_layers_.clear();
    kvmem_raw_layer_slot_.clear();
    kvmem_raw_capture_dev_.clear();
    kvmem_raw_capture_rows_ = 0;
    kvmem_raw_capture_host_.reset();
    kvmem_raw_transfer_dev_.reset();
    kvmem_raw_transfer_host_.reset();
    kvmem_raw_transfer_blocks_ = 0;
    kvmem_raw_decode_block_start_ = -1;
    kvmem_raw_decode_first_row_ = 0;
    kvmem_raw_decode_rows_ = 0;
    kvmem_raw_mtp_k_chunks_.clear();
    kvmem_raw_mtp_k_mirror_bytes_ = 0;
    kvmem_raw_mtp_k_valid_tokens_.clear();
    kvmem_raw_mtp_capture_dev_.reset();
    kvmem_raw_mtp_capture_rows_ = 0;
    kvmem_raw_mtp_capture_host_.reset();
    kvmem_raw_mtp_transfer_dev_.reset();
    kvmem_raw_mtp_transfer_host_.reset();
    kvmem_sparse_cpu_tier_ = false;
    kvmem_cpu_budget_bytes_ = 0;
    kvmem_cpu_sparse_bytes_ = 0;
    kvmem_cpu_sparse_slabs_.clear();
    kvmem_cpu_sparse_slot_live_.clear();
    kvmem_cpu_sparse_slots_per_slab_ = 1;
    kvmem_cpu_sparse_slot_bytes_ = 0;
    kvmem_defer_prefill_pressure_ = false;
    kvmem_deferred_prefill_tokens_ = 0;
    // Tear down any MTP tiering from a prior config. Release pages back to the
    // (still-live) MTP pool and detach the allocator BEFORE destroying the pool
    // so reset() never touches a freed allocator. mtp_scratch_ready_/mtp_*_cache_
    // are reset so ensure_mtp_scratch reallocates fresh (dense or bounded) on the
    // next forward; configure_kvmem always precedes any forward.
    mtp_kv_pages_.reset();
    mtp_kv_pages_.set_allocator(nullptr);
    kvmem_mtp_gpu_page_pool_.reset();
    kvmem_mtp_tiered_ = false;
    mtp_scratch_ready_ = false;
    mtp_k_cache_.reset();
    mtp_v_cache_.reset();
    mtp_baked_pos_.clear();
    const bool can_spill =
        effective.estimated_block_bytes > 0 &&
        (effective.cpu_tier_bytes > 0 || effective.nvme_tier_bytes > 0);
    if (!external_kv_cache_ && can_spill &&
        effective.estimated_gpu_block_capacity > 0) {
        const uint64_t ctx_blocks =
            (static_cast<uint64_t>(kv_ctx_size_) + effective.block_tokens - 1) /
            effective.block_tokens;
        const uint32_t gpu_blocks = std::max<uint32_t>(
            1, std::min<uint64_t>(effective.estimated_gpu_block_capacity,
                                  ctx_blocks));
        if (gpu_blocks >= 2 && gpu_blocks < ctx_blocks) {
            const uint32_t pages_per_block = std::max<uint32_t>(
                1, effective.block_tokens / std::max<uint32_t>(1, page_size));
            const uint32_t pool_pages = std::max<uint32_t>(
                1, gpu_blocks * pages_per_block);
            kvmem_gpu_page_pool_ = std::make_unique<LocalKvPagePool>(
                pool_pages, page_size, "main");
            kv_pages_.set_allocator(kvmem_gpu_page_pool_.get());
            allocate_kvmem_gpu_cache(
                static_cast<uint64_t>(pool_pages) * page_size);
            // Give the MTP layer its own bounded sibling pool so its KV rides
            // the same GPU->CPU->NVMe tiering as the 16 std layers instead of a
            // dense full-context cache. Single-request internal-MTP path only.
            if (mtp_will_tier) {
                kvmem_mtp_gpu_page_pool_ = std::make_unique<LocalKvPagePool>(
                    pool_pages, page_size, "mtp");
                mtp_kv_pages_.set_allocator(kvmem_mtp_gpu_page_pool_.get());
                kvmem_mtp_tiered_ = true;
                allocate_kvmem_mtp_gpu_cache(
                    static_cast<uint64_t>(pool_pages) * page_size);
            }
            if (kvmem_tier_trace_enabled()) {
                std::fprintf(stderr,
                             "[kvmem-tier] bounded_gpu_pool blocks=%u pages=%u page_size=%u slots=%llu mtp_tiered=%d\n",
                             gpu_blocks, pool_pages, page_size,
                             static_cast<unsigned long long>(
                                 static_cast<uint64_t>(pool_pages) * page_size),
                             kvmem_mtp_tiered_ ? 1 : 0);
            }
        }
    }
    if (kvmem_immutable_source_k_ && !kvmem_gpu_page_pool_) {
        throw std::runtime_error(
            "KVMem immutable K requires the bounded tiered GPU "
            "pool (configure CPU/NVMe spill and a context larger than the GPU "
            "resident pool)");
    }
    if (kvmem_immutable_source_k_) {
        kvmem_raw_k_block_major_ =
            effective.legacy_optimization_profile
                ? env_flag_enabled("QW3_KVMEM_RAW_K_BLOCK_MAJOR", true)
                : true;
        kvmem_raw_layers_.reserve(standard_layers);
        kvmem_raw_layer_slot_.assign(weights_.n_layers(), -1);
        for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
            if (!model_cfg.is_standard_attention_layer(il)) continue;
            kvmem_raw_layer_slot_[il] =
                static_cast<int32_t>(kvmem_raw_layers_.size());
            kvmem_raw_layers_.push_back(il);
        }
        kvmem_raw_k_row_bytes_ = raw_k_row_bytes;
        const uint64_t raw_chunk_count =
            (static_cast<uint64_t>(kv_ctx_size_) +
             kvmem_raw_k_chunk_tokens_ - 1) /
            kvmem_raw_k_chunk_tokens_;
        kvmem_raw_k_chunks_.resize(
            static_cast<size_t>(raw_chunk_count));
        kvmem_raw_k_valid_tokens_.assign(kv_ctx_size_, 0);
        if (kvmem_mtp_local_positions_) {
            kvmem_raw_mtp_k_chunks_.resize(
                static_cast<size_t>(raw_chunk_count));
            kvmem_raw_mtp_k_valid_tokens_.assign(kv_ctx_size_, 0);
        }
        if (effective.legacy_optimization_profile) {
            if (const char *env =
                    std::getenv("QW3_KVMEM_RAW_K_TRANSFER_BLOCKS")) {
                const long long v = std::atoll(env);
                if (v > 0) {
                    kvmem_raw_transfer_block_cap_ =
                        static_cast<uint32_t>(v);
                }
            }
        } else if (!effective.packed_rematerialization) {
            // The ablation keeps immutable raw K for correctness, but removes
            // the cross-block CPU pack/H2D/scatter batch. Each raw refresh is
            // therefore transferred and rematerialized one block at a time.
            kvmem_raw_transfer_block_cap_ = 1;
        }
        std::fprintf(stderr,
                     "[kvmem-tier] immutable_source_k=1 "
                     "gpu_working_k_bytes=0 raw_k_alloc=demand "
                     "raw_k_chunk_tokens=%u raw_k_layout=%s "
                     "raw_k_max_bytes=%llu "
                     "raw_mtp_k_max_bytes=%llu raw_k_allocated_bytes=0 "
                     "mtp_local_positions=%d cpu_shared_budget_bytes=%llu "
                     "cpu_spill_floor_bytes=%llu "
                     "refresh_remaps=%u refresh_tokens=%llu max_baked_pos=%llu\n",
                     kvmem_raw_k_chunk_tokens_,
                     kvmem_raw_k_block_major_
                         ? "block-major" : "layer-major",
                     static_cast<unsigned long long>(
                         raw_k_mirror_max_bytes),
                     static_cast<unsigned long long>(
                         raw_mtp_k_mirror_max_bytes),
                     kvmem_mtp_local_positions_ ? 1 : 0,
                     static_cast<unsigned long long>(
                         effective.cpu_tier_bytes),
                     static_cast<unsigned long long>(
                         capacity_cpu_tier_bytes),
                     effective.immutable_refresh_remaps,
                     static_cast<unsigned long long>(
                         effective.immutable_refresh_abs_delta_tokens),
                     static_cast<unsigned long long>(
                         effective.immutable_max_baked_position));
    }
    const bool heat_aware_cpu_cache =
        effective.legacy_optimization_profile
            ? effective.optimization_level >= KvMemOptimizationLevel::Opt1
            : effective.hierarchical_reuse;
    if (effective.cpu_tier_bytes > 0 && effective.estimated_block_bytes > 0) {
        PinnedKvTierConfig pcfg;
        pcfg.total_bytes = effective.cpu_tier_bytes;
        pcfg.slot_bytes = effective.estimated_block_bytes;
        pcfg.cache_policy =
            heat_aware_cpu_cache
                ? PinnedKvCachePolicy::HeatAware
                : PinnedKvCachePolicy::LegacyLru;
        kvmem_cpu_tier_ = std::make_unique<PinnedKvTier>(pcfg);
        if (kvmem_cpu_tier_->enabled()) {
            if (kvmem_immutable_source_k_) {
                kvmem_sparse_cpu_tier_ = true;
                kvmem_cpu_budget_bytes_ = effective.cpu_tier_bytes;
                kvmem_cpu_sparse_slot_bytes_ = pcfg.slot_bytes;
                kvmem_cpu_sparse_slots_per_slab_ =
                    static_cast<uint32_t>(std::max<uint64_t>(
                        1, kvmem_io_slab_bytes() / pcfg.slot_bytes));
                const uint32_t slot_count =
                    kvmem_cpu_tier_->slot_count();
                const uint32_t slab_count =
                    (slot_count + kvmem_cpu_sparse_slots_per_slab_ - 1) /
                    kvmem_cpu_sparse_slots_per_slab_;
                kvmem_cpu_sparse_slabs_.resize(slab_count);
                kvmem_cpu_sparse_slot_live_.assign(slot_count, 0);
                for (uint32_t slab = 0; slab < slab_count; ++slab) {
                    const uint32_t first =
                        slab * kvmem_cpu_sparse_slots_per_slab_;
                    kvmem_cpu_sparse_slabs_[slab].capacity_slots =
                        std::min(
                            kvmem_cpu_sparse_slots_per_slab_,
                            slot_count - first);
                }
                std::fprintf(
                    stderr,
                    "[kvmem-io] cpu_sparse_allocator mode=slab "
                    "slots=%u slots_per_slab=%u slab_mib=%.3f "
                    "slabs_max=%u\n",
                    slot_count,
                    kvmem_cpu_sparse_slots_per_slab_,
                    (static_cast<double>(
                         kvmem_cpu_sparse_slots_per_slab_) *
                     static_cast<double>(
                         kvmem_cpu_sparse_slot_bytes_)) /
                        (1024.0 * 1024.0),
                    slab_count);
            } else {
                const uint64_t buf_bytes =
                    static_cast<uint64_t>(kvmem_cpu_tier_->slot_count()) *
                    pcfg.slot_bytes;
                kvmem_cpu_bytes_ = host_tier_pool_
                    ? host_tier_pool_->acquire(buf_bytes, "kvmem_cpu_tier")
                    : backend_.host_buffer(buf_bytes, "kvmem_cpu_tier");
            }
        }
    }
    // A CPU-only inclusive tier is safe only when it can retain a clean record
    // for every possible context block in addition to the immutable raw-K
    // authority. Otherwise keeping CPU slots after stage-in can consume the
    // last free spill slot and make a later GPU eviction impossible. SSD-backed
    // opt_2 already has its own inclusive authority and does not use this flag.
    if (effective.optimization_level >= KvMemOptimizationLevel::Opt2 &&
        kvmem_cpu_tier_ && effective.nvme_tier_bytes == 0 &&
        effective.estimated_block_bytes > 0) {
        const uint64_t ctx_blocks =
            (static_cast<uint64_t>(kv_ctx_size_) +
             effective.block_tokens - 1) /
            effective.block_tokens;
        const uint64_t guaranteed_cpu_slots =
            capacity_cpu_tier_bytes / effective.estimated_block_bytes;
        kvmem_inclusive_cpu_backing_ =
            guaranteed_cpu_slots >= ctx_blocks &&
            (effective.legacy_optimization_profile
                 ? env_flag_enabled("QW3_KVMEM_INCLUSIVE_CPU", true)
                 : true);
    }
    if (effective.nvme_tier_bytes > 0 && effective.estimated_block_bytes > 0) {
        if (effective.nvme_tier_dir.empty()) {
            throw std::runtime_error(
                "--kvmem-nvme-bytes requires --kvmem-nvme-dir");
        }
        NvmeKvTierConfig ncfg;
        ncfg.dir = effective.nvme_tier_dir;
        ncfg.total_bytes = effective.nvme_tier_bytes;
        ncfg.slot_bytes = effective.estimated_block_bytes;
        ncfg.drop_page_cache = env_flag_enabled(
            "QW3_KVMEM_DROP_PAGE_CACHE",
            effective.optimization_level >=
                KvMemOptimizationLevel::Opt2);
        kvmem_nvme_tier_ = std::make_unique<NvmeKvTier>(std::move(ncfg));
        std::fprintf(
            stderr,
            "[kvmem-io] page_cache_policy=%s reason=%s\n",
            kvmem_nvme_tier_->drops_page_cache()
                ? "writeback-and-drop-after-io" : "kernel-default",
            kvmem_nvme_tier_->drops_page_cache()
                ? "bound-host-memory" : "compatibility-baseline");
        if (effective.optimization_level >=
            KvMemOptimizationLevel::Opt2) {
            const uint64_t warm0 = kvmem_steady_ns();
            const size_t slab_bytes = kvmem_io_slab_bytes();
            const size_t slab_count = kvmem_write_queue_depth();
            for (size_t i = 0; i < slab_count; ++i) {
                auto slab =
                    std::make_shared<std::vector<uint8_t>>();
                // resize value-initializes every page, moving allocation and
                // first-touch faults out of the reselection critical path.
                slab->resize(slab_bytes);
                slab->clear();
                kvmem_free_write_slabs_.push_back(std::move(slab));
            }
            std::fprintf(
                stderr,
                "[kvmem-io] write_slab_pool slabs=%zu slab_mib=%zu "
                "pageable_mib=%zu warmup_ms=%.3f\n",
                slab_count,
                static_cast<size_t>(
                    slab_bytes / (1024ull * 1024ull)),
                static_cast<size_t>(
                    slab_count * slab_bytes / (1024ull * 1024ull)),
                (kvmem_steady_ns() - warm0) / 1.0e6);
        }
    }
    const bool has_ssd = kvmem_nvme_tier_ != nullptr;
    const bool async_d2h =
        effective.optimization_level >= KvMemOptimizationLevel::Opt2;
    const bool async_ssd_write = async_d2h && has_ssd;
    const bool pipelined_h2d =
        effective.packed_rematerialization;
    const bool coalesced_ssd_read = pipelined_h2d && has_ssd;
    kvmem_cpu_worker_pool_.reset();
    const size_t max_cpu_workers = std::max(
        kvmem_cpu_gather_threads(), kvmem_cpu_stageout_threads());
    const bool persistent_cpu_pool =
        (async_d2h || pipelined_h2d) &&
        max_cpu_workers > 1 &&
        (effective.legacy_optimization_profile
             ? env_flag_enabled("QW3_KVMEM_PERSISTENT_CPU_POOL", true)
             : true);
    if (persistent_cpu_pool) {
        kvmem_cpu_worker_pool_ =
            std::make_unique<KvmemCpuWorkerPool>(max_cpu_workers - 1);
    }
    kvmem_stagein_worker_pool_.reset();
    kvmem_stagein_assembly_overlap_enabled_ =
        persistent_cpu_pool &&
        pipelined_h2d &&
        kvmem_immutable_source_k_ &&
        !has_ssd &&
        (effective.legacy_optimization_profile
             ? env_flag_enabled(
                   "QW3_KVMEM_OVERLAP_STAGEIN_ASSEMBLY", true)
             : true);
    if (kvmem_stagein_assembly_overlap_enabled_) {
        // The async stage-in caller is one participant. Profiling this
        // dual-socket EPYC showed that two workers leave
        // each sparse memcpy stream under-filled: restricting both sides to
        // two made the overlapped wall time exceed the original serial sum.
        // V spill gathering also remains the critical branch at 4+4, so use
        // eight V workers by default while raw K retains four. Both settings
        // are bounded and independently tunable.
        const size_t stagein_workers =
            kvmem_overlap_stagein_threads();
        if (stagein_workers > 1) {
            kvmem_stagein_worker_pool_ =
                std::make_unique<KvmemCpuWorkerPool>(
                    stagein_workers - 1);
        }
    }
    kvmem_stagein_assembly_overlap_active_ = false;
    kvmem_query_prefetch_enabled_ =
        effective.legacy_optimization_profile &&
        pipelined_h2d &&
        kvmem_inclusive_cpu_backing_ &&
        kvmem_gpu_page_pool_ != nullptr &&
        kvmem_cpu_tier_ != nullptr &&
        // Query-first-pass prefetch is advisory and workload-dependent. On the
        // 512K profile it hid its own 144 ms, but 197/410 transferred blocks
        // were unused and ordinary stage-in was already hidden by raw-K
        // assembly, so enabling it by default only increased PCIe/CPU traffic.
        env_flag_enabled("QW3_KVMEM_QUERY_PREFETCH", false);
    const bool proactive_cpu_writeback =
        async_d2h &&
        !has_ssd &&
        kvmem_inclusive_cpu_backing_;
    kvmem_prefill_writeback_enabled_ =
        effective.proactive_stage_out &&
        (async_ssd_write || proactive_cpu_writeback) &&
        kvmem_immutable_source_k_ &&
        (!kvmem_mtp_tiered_ || kvmem_mtp_local_positions_) &&
        (effective.legacy_optimization_profile
             ? env_flag_enabled("QW3_KVMEM_PREFILL_WRITEBACK", true)
             : true);
    const bool tiering_active =
        kvmem_gpu_page_pool_ != nullptr &&
        (kvmem_cpu_tier_ != nullptr || kvmem_nvme_tier_ != nullptr);
    if (!effective.legacy_optimization_profile && tiering_active) {
        if (effective.proactive_stage_out &&
            !kvmem_prefill_writeback_enabled_) {
            std::string reason = "required backing path is unavailable";
            if (!has_ssd && !kvmem_inclusive_cpu_backing_) {
                reason =
                    "CPU-only proactive stage-out requires enough CPU "
                    "capacity for an inclusive clean backing";
            } else if (kvmem_mtp_tiered_ &&
                       !kvmem_mtp_local_positions_) {
                reason =
                    "tiered MTP lacks a position-safe immutable backing";
            }
            throw std::runtime_error(
                "KVMem proactive-stage-out was requested but cannot be "
                "enabled: " + reason +
                "; fix the tier configuration or add "
                "--kvmem-optimize-off proactive-stage-out");
        }
        if (effective.packed_rematerialization &&
            (!kvmem_raw_pipeline_enabled_ ||
             std::strcmp(backend_.name(), "cuda-device") != 0)) {
            throw std::runtime_error(
                "KVMem packed-rematerialization was requested but the "
                "backend does not provide the required packed "
                "gather/H2D/GPU-scatter pipeline; add "
                "--kvmem-optimize-off packed-rematerialization");
        }
    }
    auto log_opt_status = [&](const char *name, bool requested,
                              const char *details) {
        const char *status = !requested
            ? "disabled"
            : (tiering_active ? "enabled" : "not-applicable");
        const char *reason = !requested
            ? (effective.legacy_optimization_profile
                   ? "legacy-profile" : "cli")
            : (tiering_active ? "-" : "context-fits-gpu-or-no-tier");
        std::fprintf(
            stderr,
            "[kvmem-opt-status] name=%s requested=%s status=%s "
            "reason=%s %s\n",
            name, requested ? "on" : "off", status, reason, details);
    };
    log_opt_status(
        "proactive-stage-out", effective.proactive_stage_out,
        kvmem_prefill_writeback_enabled_
            ? (has_ssd ? "target=ssd" : "target=cpu")
            : "target=none");
    log_opt_status(
        "hierarchical-reuse", effective.hierarchical_reuse,
        effective.hierarchical_reuse
            ? (kvmem_cpu_tier_
                   ? (heat_aware_cpu_cache
                          ? "gpu_delta=1 cpu_policy=heat-aware"
                          : "gpu_delta=1 cpu_policy=legacy-lru")
                   : "gpu_delta=1 cpu_policy=not-applicable")
            : "gpu_delta=0 cpu_policy=legacy-lru");
    log_opt_status(
        "packed-rematerialization",
        effective.packed_rematerialization,
        effective.packed_rematerialization
            ? (has_ssd
                   ? "cross_block_pack=1 batched_gpu_scatter=1 "
                     "ssd_coalesced=1 "
                     "assembly_pipeline=1 overlap=ssd-pipeline"
                   : (kvmem_stagein_assembly_overlap_enabled_
                          ? "cross_block_pack=1 batched_gpu_scatter=1 "
                            "ssd_coalesced=0 assembly_pipeline=1 "
                            "overlap=stagein-assembly"
                          : "cross_block_pack=1 batched_gpu_scatter=1 "
                            "ssd_coalesced=0 assembly_pipeline=1 "
                            "overlap=unavailable"))
            : "cross_block_pack=0 batched_gpu_scatter=0 "
              "raw_batch_blocks=1 ssd_coalesced=0 "
              "assembly_pipeline=0 overlap=0");
    const bool all_feature_groups_enabled =
        effective.proactive_stage_out &&
        effective.hierarchical_reuse &&
        effective.packed_rematerialization;
    std::fprintf(
        stderr,
        "[kvmem-opt] mode=%s level=%s proactive_stage_out=%d "
        "hierarchical_reuse=%d packed_rematerialization=%d "
        "cpu_cache_policy=%s "
        "async_d2h=%d inclusive_cpu=%d async_ssd_write=%d "
        "pipelined_cpu_h2d=%d coalesced_ssd_read=%d "
        "prefill_writeback=%d prefill_writeback_target=%s "
        "io_slab_mib=%zu write_qd=%zu read_qd=2 "
        "cpu_gather_threads=%zu cpu_stageout_threads=%zu "
        "persistent_cpu_pool=%d persistent_cpu_workers=%zu "
        "stagein_assembly_overlap=%d overlap_cpu_workers=%u "
        "mtp_incremental_assembly=%d query_prefetch=%d "
        "implemented_max=opt_3\n",
        effective.legacy_optimization_profile
            ? "legacy"
            : (all_feature_groups_enabled
                   ? "default-all-on" : "feature-ablation"),
        kvmem_optimization_level_name(effective.optimization_level),
        effective.proactive_stage_out ? 1 : 0,
        effective.hierarchical_reuse ? 1 : 0,
        effective.packed_rematerialization ? 1 : 0,
        heat_aware_cpu_cache
            ? "heat-aware" : "legacy-lru",
        async_d2h ? 1 : 0,
        kvmem_inclusive_cpu_backing_ ? 1 : 0,
        async_ssd_write ? 1 : 0,
        pipelined_h2d ? 1 : 0, coalesced_ssd_read ? 1 : 0,
        kvmem_prefill_writeback_enabled_ ? 1 : 0,
        kvmem_prefill_writeback_enabled_
            ? (has_ssd ? "ssd" : "cpu") : "off",
        static_cast<size_t>(
            kvmem_io_slab_bytes() / (1024ull * 1024ull)),
        kvmem_write_queue_depth(),
        kvmem_cpu_gather_threads(),
        kvmem_cpu_stageout_threads(),
        persistent_cpu_pool ? 1 : 0,
        kvmem_cpu_worker_pool_
            ? kvmem_cpu_worker_pool_->max_workers()
            : 0,
        kvmem_stagein_assembly_overlap_enabled_ ? 1 : 0,
        kvmem_stagein_assembly_overlap_enabled_
            ? static_cast<unsigned>(
                  kvmem_cpu_gather_threads() +
                  kvmem_overlap_stagein_threads())
            : 0u,
        kvmem_mtp_incremental_assembly_ ? 1 : 0,
        kvmem_query_prefetch_enabled_ ? 1 : 0);
    if (kvmem_prefill_writeback_enabled_) {
        const uint64_t warm0 = kvmem_steady_ns();
        const size_t slab_bytes = kvmem_io_slab_bytes();
        kvmem_writeback_slab_count_ =
            kvmem_writeback_slab_count();
        for (size_t i = 0;
             i < kvmem_writeback_slab_count_; ++i) {
            std::unique_ptr<HostBuffer> slab =
                backend_.host_buffer(
                    slab_bytes, "kvmem_prefill_writeback_slab");
            kvmem_free_writeback_slabs_.push_back(
                std::shared_ptr<HostBuffer>(std::move(slab)));
        }
        std::fprintf(
            stderr,
            "[kvmem-io] prefill_writeback_slab_pool slabs=%zu "
            "slab_mib=%zu pinned_mib=%zu warmup_ms=%.3f\n",
            kvmem_writeback_slab_count_,
            static_cast<size_t>(
                slab_bytes / (1024ull * 1024ull)),
            static_cast<size_t>(
                kvmem_writeback_slab_count_ * slab_bytes /
                (1024ull * 1024ull)),
            (kvmem_steady_ns() - warm0) / 1.0e6);
    }
    if (async_d2h && !has_ssd) {
        const uint64_t warm0 = kvmem_steady_ns();
        const size_t slab_bytes = kvmem_io_slab_bytes();
        constexpr size_t kStageOutSlabs = 2;
        for (size_t i = 0; i < kStageOutSlabs; ++i) {
            kvmem_free_stageout_slabs_.push_back(
                backend_.host_buffer(
                    slab_bytes, "kvmem_stageout_cpu_slab"));
        }
        std::fprintf(
            stderr,
            "[kvmem-io] cpu_stageout_slab_pool slabs=%zu slab_mib=%zu "
            "pinned_mib=%zu warmup_ms=%.3f\n",
            kStageOutSlabs,
            static_cast<size_t>(slab_bytes / (1024ull * 1024ull)),
            static_cast<size_t>(
                kStageOutSlabs * slab_bytes /
                (1024ull * 1024ull)),
            (kvmem_steady_ns() - warm0) / 1.0e6);
    }
    if (pipelined_h2d) {
        const uint64_t warm0 = kvmem_steady_ns();
        const size_t slab_bytes = kvmem_io_slab_bytes();
        // CPU hits and SSD misses have independent producers. A mixed tier
        // gets two slabs per producer so neither path repeatedly allocates
        // pinned memory after the other has consumed the common pool.
        kvmem_read_slab_count_ =
            has_ssd && kvmem_cpu_tier_ ? 4 : 2;
        for (size_t i = 0; i < kvmem_read_slab_count_; ++i) {
            kvmem_free_read_slabs_.push_back(
                backend_.host_buffer(
                    slab_bytes, "kvmem_stagein_read_slab"));
        }
        std::fprintf(
            stderr,
            "[kvmem-io] read_slab_pool slabs=%zu slab_mib=%zu "
            "pinned_mib=%zu warmup_ms=%.3f\n",
            kvmem_read_slab_count_,
            static_cast<size_t>(
                slab_bytes / (1024ull * 1024ull)),
            static_cast<size_t>(
                kvmem_read_slab_count_ * slab_bytes /
                (1024ull * 1024ull)),
            (kvmem_steady_ns() - warm0) / 1.0e6);
    }
    // Capacity guarantee for the bounded GPU pool. When the pool engaged
    // (gpu_blocks < ctx_blocks, so a >budget prompt must spill), every history
    // block beyond the resident selection budget has to live in a CPU/NVMe
    // spill tier. Selection keeps at most budget_blocks GPU-resident, so a
    // prompt filling --ctx pushes up to (ctx_blocks - budget_blocks) blocks
    // into the tiers at once. If CPU+NVMe cannot hold that many, a long
    // query-conditioned prefill fills the pool with blocks that have nowhere
    // to spill and throws "local KVMem GPU page pool exhausted" mid-request
    // (kvmem_stage_out can neither place nor release them). Fail fast here,
    // at configure time, naming the exact knobs to change.
    if (kvmem_gpu_page_pool_ && effective.estimated_block_bytes > 0) {
        const uint64_t cap_ctx_blocks =
            (static_cast<uint64_t>(kv_ctx_size_) + effective.block_tokens - 1) /
            effective.block_tokens;
        const uint64_t cap_budget_blocks = std::max<uint64_t>(
            1, (static_cast<uint64_t>(effective.select_budget) +
                effective.block_tokens - 1) / effective.block_tokens);
        const uint64_t spill_needed_blocks =
            cap_ctx_blocks > cap_budget_blocks
                ? cap_ctx_blocks - cap_budget_blocks
                : 0;
        const uint64_t cpu_slots =
            effective.estimated_block_bytes > 0
                ? capacity_cpu_tier_bytes /
                      effective.estimated_block_bytes
                : 0;
        const uint64_t nvme_slots =
            kvmem_nvme_tier_ ? kvmem_nvme_tier_->slot_count() : 0;
        const bool inclusive_ssd =
            effective.optimization_level >= KvMemOptimizationLevel::Opt2 &&
            kvmem_nvme_tier_ != nullptr;
        const bool inclusive =
            inclusive_ssd || kvmem_inclusive_cpu_backing_;
        const uint64_t required_blocks =
            inclusive ? cap_ctx_blocks : spill_needed_blocks;
        const uint64_t available_blocks =
            inclusive_ssd ? nvme_slots
                          : kvmem_inclusive_cpu_backing_
                                ? cpu_slots
                                : cpu_slots + nvme_slots;
        if (available_blocks < required_blocks) {
            auto gib = [&](uint64_t blocks) {
                char buf[32];
                const double v =
                    static_cast<double>(blocks) *
                    static_cast<double>(effective.estimated_block_bytes) /
                    (1024.0 * 1024.0 * 1024.0);
                std::snprintf(buf, sizeof(buf), "%.2f", v);
                return std::string(buf);
            };
            if (inclusive) {
                throw std::runtime_error(
                    "kvmem: opt_2 inclusive backing is too small. A "
                    "prompt near --ctx can create " +
                    std::to_string(cap_ctx_blocks) + " blocks (" +
                    gib(cap_ctx_blocks) +
                    " GiB of spill records), but the authoritative backing "
                    "provides only " +
                    std::to_string(available_blocks) + " slots (" +
                    gib(nvme_slots) +
                    " GiB). Raise --kvmem-nvme-gb or lower --ctx.");
            }
            throw std::runtime_error(
                "kvmem: spill tiers too small for the bounded GPU pool. A prompt "
                "near --ctx (" + std::to_string(cap_ctx_blocks) + " blocks) would "
                "spill up to " + std::to_string(spill_needed_blocks) + " blocks (" +
                gib(spill_needed_blocks) + " GiB) into CPU+NVMe beyond the " +
                std::to_string(cap_budget_blocks) + "-block GPU selection budget, "
                "but only " + std::to_string(available_blocks) + " spill slots (" +
                gib(available_blocks) + " GiB: cpu=" + std::to_string(cpu_slots) +
                " + nvme=" + std::to_string(nvme_slots) + ") are configured. A long "
                "query-conditioned prefill would exhaust the GPU page pool. Raise "
                "--kvmem-cpu-gb and/or add --kvmem-nvme-dir + --kvmem-nvme-gb, or "
                "lower --ctx, or raise --kvmem-budget.");
        }
    }
    bs_score_ready_ = false;
    bs_window_blocks_ = 0;
    bs_window_block_ids_.clear();
    bs_win_base_host_.clear();
    bs_blk_tokens_host_.clear();
    g_content_ready_ = false;
    g_query_ready_ = false;
    g_indexed_blocks_ = 0;
    g_orig_base_host_.clear();
    g_blk_tokens_host_.clear();
}

void QwenExecutor::kvmem_register_append(uint32_t n_new_tokens) {
    if (!kvmem_enabled_ || !block_store_) return;
    if (n_new_tokens == 0) return;
    if (position_ <= kvmem_registered_pos_) return;
    const uint32_t append_begin = kvmem_registered_pos_;
    const uint32_t target = std::min<uint32_t>(
        position_, kvmem_registered_pos_ + n_new_tokens);
    kvmem_register_until(target);

    // Decode/MTP callers register only after the accepted model rows have
    // advanced both position_ and the active window tail.  Newly-created blocks
    // therefore need their actual window construction frame recorded here;
    // register_append's default orig_pos_start is only correct in the dense
    // identity frame.
    if (kvmem_active_ && (kvmem_fix_bakedpos_ || kvmem_immutable_source_k_)) {
        const int64_t delta = static_cast<int64_t>(window_query_pos_) -
                              static_cast<int64_t>(position_);
        if (delta != 0) {
            const auto &blocks = block_store_->blocks();
            for (uint32_t bi = static_cast<uint32_t>(blocks.size());
                 bi-- > 0;) {
                const KvMemBlock &b = blocks[bi];
                if (b.orig_pos_start < append_begin) break;
                block_store_->set_block_baked_pos(
                    bi, static_cast<int64_t>(b.orig_pos_start) + delta);
            }
        }
    }
}

void QwenExecutor::kvmem_register_until(uint32_t target_pos) {
    if (!kvmem_enabled_ || !block_store_) return;
    if (target_pos <= kvmem_registered_pos_) return;
    block_store_->register_append(target_pos - kvmem_registered_pos_);
    kvmem_registered_pos_ = target_pos;
}

void QwenExecutor::kvmem_truncate_to(uint32_t token_pos) {
    if (!kvmem_enabled_ || !block_store_) return;
    kvmem_finish_proactive_d2h(/*wait_all=*/true);
    kvmem_reap_pending_writes(/*wait_all=*/true);
    // Rewind the block table to `token_pos` tokens. restore_state() has already
    // rewound position/KV-pages/recurrent/window; this reconciles the block
    // store + tier slots + selection indices, which restore_state does not touch.
    std::vector<KvMemDroppedBlock> dropped = block_store_->truncate_to(token_pos);
    kvmem_writeback_next_block_ = std::min<uint32_t>(
        kvmem_writeback_next_block_, block_store_->block_count());
    if (token_pos < kvmem_raw_k_valid_tokens_.size()) {
        std::fill(kvmem_raw_k_valid_tokens_.begin() + token_pos,
                  kvmem_raw_k_valid_tokens_.end(),
                  static_cast<uint8_t>(0));
    }
    if (token_pos < kvmem_raw_mtp_k_valid_tokens_.size()) {
        std::fill(kvmem_raw_mtp_k_valid_tokens_.begin() + token_pos,
                  kvmem_raw_mtp_k_valid_tokens_.end(),
                  static_cast<uint8_t>(0));
    }
    kvmem_truncate_raw_k(token_pos);
    kvmem_raw_decode_block_start_ = -1;
    kvmem_raw_decode_first_row_ = 0;
    kvmem_raw_decode_rows_ = 0;
    // Release CPU/NVMe tier slots for dropped blocks. GPU pages are NOT released
    // here: restore_state's truncate_to_logical_pages already freed the physical
    // pages for positions >= token_pos, so a second release would double-free.
    for (const KvMemDroppedBlock &d : dropped) {
        if (d.cpu_slot >= 0 && kvmem_cpu_tier_) {
            kvmem_cpu_tier_->release_block(d.block_id);
            kvmem_release_cpu_slot(d.cpu_slot);
        }
        if (d.nvme_slot >= 0 && kvmem_nvme_tier_) kvmem_nvme_tier_->release_block(d.block_id);
    }
    // Invalidate the per-session selection indices so a stale index over the old
    // (larger) block count cannot survive. The warm-resume pressure selection and
    // the post-suffix semantic selection rebuild over the corrected block table.
    // Position/KV/recurrent/window are NOT touched here.
    bs_score_ready_ = false;
    bs_window_blocks_ = 0;
    bs_window_block_ids_.clear();
    bs_win_base_host_.clear();
    bs_blk_tokens_host_.clear();
    g_content_ready_ = false;
    g_query_ready_ = false;
    g_indexed_blocks_ = 0;
    g_orig_base_host_.clear();
    g_blk_tokens_host_.clear();
    g_kbar_multi_ready_ = false;
    g_kbar_multi_blocks_ = 0;
    kvmem_qc_total_blocks_ = 0;
    kvmem_qc_prompt_tokens_ = 0;
    kvmem_qc_captured_blocks_ = 0;
    kvmem_qc_captured_tokens_ = 0;
    kvmem_pending_reselect_ = false;
    kvmem_pending_plan_ = KvMemPlan{};
    // Server-side session continuation: signal the reuse point to the next
    // kvmem_set_query_span so it PRESERVES the [0,token_pos) content-index slices
    // (position-invariant, fixed stride) and seeds the captured-block count instead
    // of zeroing the whole index. Set uniformly: the fixed stride keeps surviving
    // [0, floor(token_pos/bt)) slices at a stable offset whether or not high blocks
    // were dropped (divergence), and set_query_span only consumes it when a QC span
    // is active with resume_base <= prompt_tokens (cold/below-budget turns clear it
    // unused). The boundary block containing token_pos keeps its prior (possibly
    // stale) mean — a bounded ranking approximation; KV bytes stay exact.
    kvmem_qc_resume_base_tokens_ = token_pos;
    // Defensive query-span reset (belt-and-suspenders, plan §4): the resumed turn
    // re-sets the span via kvmem_set_query_span against the NEW question, but clear
    // here regardless so a stale span can never survive a truncate. Costs nothing on
    // the reset path (set_query_span overwrites these immediately).
    kvmem_drain_query_capture();
    kvmem_query_begin_ = 0;
    kvmem_query_end_ = 0;
    g_query_multi_ready_ = false;
    g_query_multi_count_ = 0;
}

void QwenExecutor::kvmem_maybe_prefill_offload(uint32_t next_chunk_tokens) {
    if (!kvmem_enabled_ || !block_store_ || !kvmem_gpu_page_pool_) return;
    // Query replay runs into a fixed, already-final semantic context window.
    // Its old suffix blocks were removed before replay, leaving the same pool
    // headroom for the replacement suffix; a pressure reselect here would
    // silently replace that fixed selection with sink+recent and invalidate the
    // controlled experiment.
    if (kvmem_query_replay_active_ || kvmem_query_prefetch_active_) return;
    if (!kvmem_cpu_tier_ && !kvmem_nvme_tier_) return;
    const KvMemStoreConfig &cfg = block_store_->config();
    if (cfg.estimated_block_bytes == 0 || cfg.block_tokens == 0) return;
    const uint32_t page_size = std::max<uint32_t>(1, kv_pages_.page_size);
    const uint32_t pages_per_block =
        std::max<uint32_t>(1, cfg.block_tokens / page_size);
    const uint32_t free_pages = kvmem_gpu_page_pool_->free_pages();
    const uint32_t keep_free_blocks =
        std::max<uint32_t>(1, cfg.recent_blocks > 0 ? cfg.recent_blocks : 1);
    // Reserve room for the next prefill append plus a recent-block cushion. The
    // append grabs ceil(next_chunk_tokens / page_size) pages in one shot before
    // any offload runs again, so offloading only once `free_pages` is down to the
    // cushion would let that append exhaust the pool and throw. Trigger early
    // enough that the upcoming chunk still fits.
    const uint32_t next_chunk_pages =
        (next_chunk_tokens + page_size - 1) / page_size;
    const uint32_t headroom_pages =
        next_chunk_pages + keep_free_blocks * pages_per_block;
    if (free_pages > headroom_pages) return;
    if (kvmem_keep_selected_prefill_) {
        throw std::runtime_error(
            "KVMem keep-selected prefill exhausted the reserved GPU headroom; "
            "reduce the appended session or increase the generation reserve");
    }
    kvmem_reselect_prefill_pressure();
}

void QwenExecutor::kvmem_set_defer_prefill_pressure(bool enabled) {
    kvmem_defer_prefill_pressure_ = enabled;
    if (!enabled && kvmem_deferred_prefill_tokens_ > 0) {
        kvmem_finish_deferred_prefill_pressure();
    }
}

void QwenExecutor::kvmem_finish_deferred_prefill_pressure() {
    const uint32_t pending = kvmem_deferred_prefill_tokens_;
    kvmem_deferred_prefill_tokens_ = 0;
    if (pending > 0) kvmem_maybe_prefill_offload(pending);
}

void QwenExecutor::sync_window_pages_device(uint32_t have_pages) {
    if (have_pages == 0) return;
    if (!window_pages_device_) {
        window_pages_device_ = backend_.tensor_i32(
            std::max<uint32_t>(kv_pages_.max_pages, 1), "kv_window_page_indices");
    }
    // The window page list is rewritten wholesale on every assembly (it is a
    // reordering, not an append), so re-upload all `have_pages` entries.
    require_status(backend_.copy_i32_from_host(
        *window_pages_device_, 0, window_pages_host_.data(), have_pages));
}

void QwenExecutor::sync_mtp_window_pages_device(uint32_t have_pages) {
    if (have_pages == 0) return;
    if (!mtp_window_pages_device_) {
        mtp_window_pages_device_ = backend_.tensor_i32(
            std::max<uint32_t>(mtp_kv_pages_.max_pages, 1),
            "mtp_kv_window_page_indices");
    }
    require_status(backend_.copy_i32_from_host(
        *mtp_window_pages_device_, 0, mtp_window_pages_host_.data(), have_pages));
}

uint64_t QwenExecutor::kvmem_kv_page_bytes() const {
    const QwenConfig &cfg = model_.config();
    const uint64_t per_pos =
        static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
    const uint64_t elems = static_cast<uint64_t>(kv_pages_.page_size) * per_pos;
    const DeviceTensor *sample = nullptr;
    for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
        if (!cfg.is_standard_attention_layer(il)) continue;
        sample = external_kv_cache_ ? external_kv_cache_->k_cache[il]
                                    : k_cache_[il].get();
        break;
    }
    if (!sample) return 0;
    // q8 is rejected at immutable-mode configuration because of row scales;
    // the remaining one-byte cache type is fp8, whose raw bytes can be tiered
    // exactly like fp16/fp32.
    return elems * static_cast<uint64_t>(sample->elem_size);
}

bool QwenExecutor::kvmem_block_pages_resident(const KvMemBlock &block) const {
    if (block.n_tokens == 0) return true;
    const uint32_t page_size = kv_pages_.page_size;
    const uint32_t first_page = block.orig_pos_start / page_size;
    const uint32_t last_page = (block.orig_pos_start + block.n_tokens - 1) / page_size;
    for (uint32_t lp = first_page; lp <= last_page; ++lp) {
        if (!kv_pages_.logical_page_resident(lp)) return false;
    }
    return true;
}

bool QwenExecutor::kvmem_block_mtp_pages_resident(const KvMemBlock &block) const {
    if (block.n_tokens == 0) return true;
    const uint32_t page_size = mtp_kv_pages_.page_size;
    const uint32_t first_page = block.orig_pos_start / page_size;
    const uint32_t last_page = (block.orig_pos_start + block.n_tokens - 1) / page_size;
    for (uint32_t lp = first_page; lp <= last_page; ++lp) {
        if (lp >= mtp_kv_pages_.pages.size() ||
            !mtp_kv_pages_.logical_page_resident(lp)) {
            return false;
        }
    }
    return true;
}

bool QwenExecutor::kvmem_mtp_prefix_covers_registered() const {
    return kvmem_mtp_tiered_ && mtp_scratch_ready_ &&
           mtp_prefix_len_ >= kvmem_registered_pos_;
}

void QwenExecutor::kvmem_sync_mtp_baked_pos() {
    if (!block_store_) return;
    const auto &blocks = block_store_->blocks();
    const size_t n = blocks.size();
    if (mtp_baked_pos_.size() >= n) return;
    const size_t old = mtp_baked_pos_.size();
    mtp_baked_pos_.resize(n);
    for (size_t i = old; i < n; ++i) {
        mtp_baked_pos_[i] = static_cast<int64_t>(blocks[i].orig_pos_start);
    }
}

uint64_t QwenExecutor::kvmem_block_spill_bytes(const KvMemBlock &block) const {
    if (block.n_tokens == 0) return 0;
    const QwenConfig &cfg = model_.config();
    const uint64_t page_bytes = kvmem_kv_page_bytes();
    const uint32_t page_size = kv_pages_.page_size;
    const uint32_t first_page = block.orig_pos_start / page_size;
    const uint32_t last_page =
        (block.orig_pos_start + block.n_tokens - 1) / page_size;
    const uint32_t n_pages = last_page - first_page + 1;
    const uint64_t standard_layers = static_cast<uint64_t>(
        count_standard_attention_layers(cfg, weights_.n_layers()));
    const uint64_t standard_copies =
        standard_layers * (kvmem_immutable_source_k_ ? 1ull : 2ull);
    const uint64_t mtp_copies = kvmem_mtp_tiered_
        ? (kvmem_mtp_local_positions_ ? 1ull : 2ull)
        : 0ull;
    return page_bytes * n_pages * (standard_copies + mtp_copies);
}

uint8_t *QwenExecutor::kvmem_cpu_data() {
    return kvmem_cpu_bytes_
        ? static_cast<uint8_t *>(kvmem_cpu_bytes_->data)
        : nullptr;
}

const uint8_t *QwenExecutor::kvmem_cpu_data() const {
    return kvmem_cpu_bytes_
        ? static_cast<const uint8_t *>(kvmem_cpu_bytes_->data)
        : nullptr;
}

uint64_t QwenExecutor::kvmem_cpu_bytes() const {
    return kvmem_cpu_bytes_ ? kvmem_cpu_bytes_->bytes : 0;
}

void QwenExecutor::kvmem_canonicalize_block_for_tier(uint32_t block_id) {
    if (!block_store_) return;
    const auto &blocks = block_store_->blocks();
    if (block_id >= blocks.size()) return;
    const KvMemBlock &block = blocks[block_id];
    if (block.n_tokens == 0) return;
    if (!kvmem_block_pages_resident(block)) {
        throw std::runtime_error(
            "KVMem canonicalize requested for a non-resident block");
    }
    const int64_t canonical = static_cast<int64_t>(block.orig_pos_start);
    const QwenConfig &cfg = model_.config();
    const uint32_t per_pos =
        static_cast<uint32_t>(cfg.n_kv_heads) * cfg.head_dim;
    // Main standard-attention layers: de-rotate from block.baked_pos (which
    // tracks the main cache's window slot) back to the block's true position so
    // the spilled bytes are position-canonical.
    if (!kvmem_immutable_source_k_ && block.baked_pos != canonical) {
        const int64_t from = block.baked_pos;
        const uint32_t standard_layers =
            count_standard_attention_layers(cfg, weights_.n_layers());
        trace_rope_position_if_out_of_range(
            "kvmem_tier_canonicalize.main.from", from, block.n_tokens,
            cfg.n_ctx_train, -1, standard_layers);
        trace_rope_position_if_out_of_range(
            "kvmem_tier_canonicalize.main.to", canonical, block.n_tokens,
            cfg.n_ctx_train, -1, standard_layers);
        for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
            if (!cfg.is_standard_attention_layer(il)) continue;
            DeviceTensor &kc = k_cache(il);
            require_status(backend_.rope_block_remap_paged_device(
                kc, block.n_tokens, cfg.n_kv_heads, per_pos, cfg.head_dim,
                cfg.rope_dim, /*win_base=*/block.orig_pos_start,
                /*from_base=*/static_cast<int32_t>(from),
                /*to_base=*/static_cast<int32_t>(block.orig_pos_start),
                kv_page_indices_device(), kv_pages_.page_size, cfg.rope_theta));
        }
        block_store_->record_block_rerope(block_id, canonical);
        if (kvmem_tier_trace_enabled()) {
            std::fprintf(stderr,
                         "[kvmem-tier] canonicalize block=%u from=%lld to=%u\n",
                         block_id, static_cast<long long>(from),
                         block.orig_pos_start);
        }
    }
    // Legacy MTP head: de-rotate from its OWN baked position. During prefill
    // main K is packed to window slots while MTP stays canonical, so
    // main.baked_pos and the MTP bake diverge; using it here would double-rotate
    // an already-canonical MTP K. mtp_baked_pos_ is the MTP-specific truth.
    // Local-position mode never canonicalizes a rotated MTP K: its raw CPU
    // mirror is authoritative and only V is spilled below.
    if (kvmem_mtp_tiered_ && !kvmem_mtp_local_positions_) {
        kvmem_sync_mtp_baked_pos();
        const int64_t mtp_from = mtp_baked_pos_[block_id];
        if (mtp_from != canonical) {
            if (kvmem_block_mtp_pages_resident(block)) {
                trace_rope_position_if_out_of_range(
                    "kvmem_tier_canonicalize.mtp.from", mtp_from,
                    block.n_tokens, cfg.n_ctx_train);
                trace_rope_position_if_out_of_range(
                    "kvmem_tier_canonicalize.mtp.to", canonical,
                    block.n_tokens, cfg.n_ctx_train);
                require_status(backend_.rope_block_remap_paged_device(
                    mtp_k_cache(), block.n_tokens, cfg.n_kv_heads, per_pos,
                    cfg.head_dim, cfg.rope_dim,
                    /*win_base=*/block.orig_pos_start,
                    /*from_base=*/static_cast<int32_t>(mtp_from),
                    /*to_base=*/static_cast<int32_t>(block.orig_pos_start),
                    mtp_kv_pages_.device_indices(), mtp_kv_pages_.page_size,
                    cfg.rope_theta));
            }
            mtp_baked_pos_[block_id] = canonical;
            if (kvmem_tier_trace_enabled()) {
                std::fprintf(stderr,
                             "[kvmem-tier] canonicalize-mtp block=%u from=%lld to=%u\n",
                             block_id, static_cast<long long>(mtp_from),
                             block.orig_pos_start);
            }
        }
    }
}

void QwenExecutor::kvmem_copy_block_to_host(const KvMemBlock &block,
                                            std::vector<uint8_t> &dst) {
    if (block.n_tokens == 0) {
        dst.clear();
        return;
    }
    dst.clear();
    dst.resize(kvmem_block_spill_bytes(block));
    kvmem_copy_block_to_host_ptr(block, dst.data());
}

void QwenExecutor::kvmem_copy_block_to_host_ptr(const KvMemBlock &block,
                                                uint8_t *out) {
    if (block.n_tokens == 0) return;
    const QwenConfig &cfg = model_.config();
    const uint64_t page_bytes = kvmem_kv_page_bytes();
    const uint32_t page_size = kv_pages_.page_size;
    const uint32_t first_page = block.orig_pos_start / page_size;
    const uint32_t last_page = (block.orig_pos_start + block.n_tokens - 1) / page_size;
    const uint64_t per_pos =
        static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
    for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
        if (!cfg.is_standard_attention_layer(il)) continue;
        DeviceTensor &kc = k_cache(il);
        DeviceTensor &vc = v_cache(il);
        if (kc.elem_size != vc.elem_size) {
            throw std::runtime_error("KVMem CPU offload found mixed K/V dtypes");
        }
        for (uint32_t lp = first_page; lp <= last_page; ++lp) {
            if (!kv_pages_.logical_page_resident(lp)) {
                throw std::runtime_error("KVMem CPU offload saw non-resident GPU page");
            }
            const int32_t physical_page = kv_pages_.pages[lp];
            const uint64_t byte_offset =
                static_cast<uint64_t>(physical_page) * page_size * per_pos *
                static_cast<uint64_t>(kc.elem_size);
            if (!kvmem_immutable_source_k_) {
                require_status(backend_.copy_bytes_to_host_async(
                    kc, out, byte_offset, page_bytes));
                out += page_bytes;
            }
            require_status(backend_.copy_bytes_to_host_async(vc, out, byte_offset,
                                                             page_bytes));
            out += page_bytes;
        }
    }
    // Trailing MTP segment. In legacy mode this is K then V per page. With
    // local-position MTP, K is never spilled: the position-free CPU mirror is
    // authoritative and assemble materializes it directly at the selected
    // compact positions, so the tier record contains V only.
    if (kvmem_mtp_tiered_) {
        DeviceTensor &mk = mtp_k_cache();
        DeviceTensor &mv = mtp_v_cache();
        if (mk.elem_size != mv.elem_size) {
            throw std::runtime_error("KVMem MTP offload found mixed K/V dtypes");
        }
        for (uint32_t lp = first_page; lp <= last_page; ++lp) {
            if (lp < mtp_kv_pages_.pages.size() &&
                mtp_kv_pages_.logical_page_resident(lp)) {
                const int32_t physical_page = mtp_kv_pages_.pages[lp];
                const uint64_t byte_offset =
                    static_cast<uint64_t>(physical_page) * page_size * per_pos *
                    static_cast<uint64_t>(mk.elem_size);
                if (!kvmem_mtp_local_positions_) {
                    require_status(backend_.copy_bytes_to_host_async(
                        mk, out, byte_offset, page_bytes));
                    out += page_bytes;
                }
                require_status(backend_.copy_bytes_to_host_async(
                    mv, out, byte_offset, page_bytes));
                out += page_bytes;
            } else {
                if (kvmem_mtp_local_positions_) {
                    throw std::runtime_error(
                        "KVMem local-position MTP attempted to spill a block "
                        "before its V page was primed");
                }
                if (!kvmem_mtp_local_positions_) {
                    std::memset(out, 0, page_bytes);
                    out += page_bytes;
                }
                std::memset(out, 0, page_bytes);
                out += page_bytes;
            }
        }
    }
}

uint8_t *QwenExecutor::kvmem_ensure_stage_pinned(uint64_t bytes) {
    if (bytes == 0) return nullptr;
    if (!kvmem_stage_pinned_ || kvmem_stage_pinned_->bytes < bytes) {
        kvmem_stage_pinned_ = backend_.host_buffer(bytes, "kvmem_stageout");
    }
    return static_cast<uint8_t *>(kvmem_stage_pinned_->data);
}

void QwenExecutor::kvmem_copy_block_from_host(
        const KvMemBlock &block, const std::vector<uint8_t> &src,
        bool stage_mtp_pages) {
    kvmem_copy_block_from_host(block, src.data(), src.size(),
                               stage_mtp_pages);
}

void QwenExecutor::kvmem_copy_block_from_host(
        const KvMemBlock &block, const void *src, uint64_t src_bytes,
        bool stage_mtp_pages) {
    if (block.n_tokens == 0) return;
    const QwenConfig &cfg = model_.config();
    const uint64_t page_bytes = kvmem_kv_page_bytes();
    const uint32_t page_size = kv_pages_.page_size;
    const uint32_t first_page = block.orig_pos_start / page_size;
    const uint32_t last_page = (block.orig_pos_start + block.n_tokens - 1) / page_size;
    const uint64_t expected = kvmem_block_spill_bytes(block);
    if (src_bytes < expected) {
        throw std::runtime_error("KVMem CPU stage-in found wrong spill buffer size");
    }
    if (!src) {
        throw std::runtime_error("KVMem CPU stage-in found null spill buffer");
    }
    const uint64_t per_pos =
        static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
    const uint8_t *in = static_cast<const uint8_t *>(src);
    for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
        if (!cfg.is_standard_attention_layer(il)) continue;
        DeviceTensor &kc = k_cache(il);
        DeviceTensor &vc = v_cache(il);
        if (kc.elem_size != vc.elem_size) {
            throw std::runtime_error("KVMem CPU stage-in found mixed K/V dtypes");
        }
        for (uint32_t lp = first_page; lp <= last_page; ++lp) {
            if (!kv_pages_.logical_page_resident(lp)) {
                throw std::runtime_error("KVMem CPU stage-in target page missing");
            }
            const int32_t physical_page = kv_pages_.pages[lp];
            const uint64_t byte_offset =
                static_cast<uint64_t>(physical_page) * page_size * per_pos *
                static_cast<uint64_t>(kc.elem_size);
            if (!kvmem_immutable_source_k_) {
                require_status(backend_.copy_bytes_from_host_async(
                    kc, byte_offset, in, page_bytes));
                in += page_bytes;
            }
            require_status(backend_.copy_bytes_from_host_async(vc, byte_offset, in,
                                                               page_bytes));
            in += page_bytes;
        }
    }
    // Trailing MTP segment: legacy records contain K then V; local-position
    // records contain V only. Raw MTP K is materialized after stage-in, directly
    // into the selected window frame.
    if (stage_mtp_pages) {
        DeviceTensor &mk = mtp_k_cache();
        DeviceTensor &mv = mtp_v_cache();
        if (mk.elem_size != mv.elem_size) {
            throw std::runtime_error("KVMem MTP stage-in found mixed K/V dtypes");
        }
        for (uint32_t lp = first_page; lp <= last_page; ++lp) {
            if (lp >= mtp_kv_pages_.pages.size() ||
                !mtp_kv_pages_.logical_page_resident(lp)) {
                throw std::runtime_error("KVMem MTP stage-in target page missing");
            }
            const int32_t physical_page = mtp_kv_pages_.pages[lp];
            const uint64_t byte_offset =
                static_cast<uint64_t>(physical_page) * page_size * per_pos *
                static_cast<uint64_t>(mk.elem_size);
            if (!kvmem_mtp_local_positions_) {
                require_status(backend_.copy_bytes_from_host_async(
                    mk, byte_offset, in, page_bytes));
                in += page_bytes;
            }
            require_status(backend_.copy_bytes_from_host_async(mv, byte_offset, in,
                                                               page_bytes));
            in += page_bytes;
        }
    }
}

void QwenExecutor::kvmem_stage_in(const KvMemPlan &plan) {
    kvmem_start_prefetch(plan);
    kvmem_finish_prefetch();
}

void QwenExecutor::kvmem_submit_prefetch_cpu_batches() {
    if (!kvmem_prefetch_.bulk_cpu || !kvmem_cpu_tier_ ||
        !block_store_) {
        return;
    }
    constexpr size_t kMaxInflightCpuBatches = 2;
    const uint64_t slab_target_bytes = kvmem_io_slab_bytes();
    const bool perf_trace = kvmem_perf_trace_flag();
    const bool stage_mtp_pages = kvmem_prefetch_.stage_mtp_pages;
    const auto &blocks = block_store_->blocks();
    while (kvmem_prefetch_.cpu_batches.size() <
               kMaxInflightCpuBatches &&
           kvmem_prefetch_.next_cpu_read <
               kvmem_prefetch_.cpu_reads.size()) {
        const size_t begin = kvmem_prefetch_.next_cpu_read;
        size_t end = begin;
        uint64_t batch_bytes = 0;
        while (end < kvmem_prefetch_.cpu_reads.size()) {
            const uint64_t bytes =
                kvmem_prefetch_.cpu_reads[end].bytes;
            if (end > begin &&
                batch_bytes + bytes > slab_target_bytes) {
                break;
            }
            kvmem_prefetch_.cpu_reads[end].batch_offset = batch_bytes;
            batch_bytes += bytes;
            ++end;
        }

        KvMemPrefetchCpuBatch batch;
        batch.read_begin = begin;
        batch.read_end = end;
        for (auto it = kvmem_free_read_slabs_.begin();
             it != kvmem_free_read_slabs_.end(); ++it) {
            if ((*it)->bytes >= batch_bytes) {
                batch.buffer = std::move(*it);
                kvmem_free_read_slabs_.erase(it);
                break;
            }
        }
        if (!batch.buffer) {
            batch.buffer = backend_.host_buffer(
                std::max<uint64_t>(batch_bytes, slab_target_bytes),
                "kvmem_stagein_cpu_slab");
        }

        const uint64_t gather0 =
            perf_trace ? kvmem_steady_ns() : 0;
        auto *slab =
            static_cast<uint8_t *>(batch.buffer->data);
        // The immutable CPU tier is intentionally pageable and sparse. A
        // selected I/O slab therefore consists of many independent block
        // allocations. One memcpy stream reached only ~3 GiB/s on the profiled
        // dual-socket host, becoming slower than PCIe H2D. Copy disjoint block
        // ranges with a small bounded worker set; H2D of the previous slab
        // continues concurrently on the CUDA copy stream.
        const size_t read_count = end - begin;
        const size_t worker_count = std::min(
            kvmem_stagein_assembly_overlap_active_
                ? kvmem_overlap_stagein_threads()
                : kvmem_cpu_gather_threads(),
            read_count);
        KvmemCpuWorkerPool *stagein_pool =
            kvmem_stagein_assembly_overlap_active_ &&
                    kvmem_stagein_worker_pool_
                ? kvmem_stagein_worker_pool_.get()
                : kvmem_cpu_worker_pool_.get();
        for (size_t i = begin; i < end; ++i) {
            if (!kvmem_cpu_slot_data(
                    kvmem_prefetch_.cpu_reads[i].slot)) {
                throw std::runtime_error(
                    "KVMem pipelined CPU stage-in lost its source slot");
            }
        }
        auto gather_range = [&](size_t first, size_t last) {
            for (size_t i = first; i < last; ++i) {
                const KvMemPrefetchCpuRead &read =
                    kvmem_prefetch_.cpu_reads[i];
                const uint8_t *src =
                    kvmem_cpu_slot_data(read.slot);
                std::memcpy(
                    slab + read.batch_offset, src, read.bytes);
            }
        };
        if (worker_count <= 1) {
            gather_range(begin, end);
        } else if (stagein_pool) {
            stagein_pool->run(
                worker_count, worker_count,
                [&](size_t worker) {
                    const size_t first =
                        begin + read_count * worker / worker_count;
                    const size_t last =
                        begin + read_count * (worker + 1) / worker_count;
                    gather_range(first, last);
                });
        } else {
            std::vector<std::thread> workers;
            workers.reserve(worker_count);
            for (size_t w = 0; w < worker_count; ++w) {
                const size_t first =
                    begin + read_count * w / worker_count;
                const size_t last =
                    begin + read_count * (w + 1) / worker_count;
                workers.emplace_back(gather_range, first, last);
            }
            for (std::thread &worker : workers) worker.join();
        }
        if (perf_trace) {
            kvmem_prefetch_.perf.cpu_gather_ns +=
                kvmem_steady_ns() - gather0;
        }

        const uint64_t enqueue0 =
            perf_trace ? kvmem_steady_ns() : 0;
        struct PackedTarget {
            DeviceTensor *tensor = nullptr;
            bool mtp = false;
        };
        std::vector<PackedTarget> packed_targets;
        packed_targets.reserve(kvmem_raw_layers_.size() * 2 + 2);
        const QwenConfig &cfg = model_.config();
        for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
            if (!cfg.is_standard_attention_layer(il)) continue;
            if (!kvmem_immutable_source_k_) {
                packed_targets.push_back(
                    PackedTarget{&k_cache(il), false});
            }
            packed_targets.push_back(
                PackedTarget{&v_cache(il), false});
        }
        if (stage_mtp_pages) {
            if (!kvmem_mtp_local_positions_) {
                packed_targets.push_back(
                    PackedTarget{&mtp_k_cache(), true});
            }
            packed_targets.push_back(
                PackedTarget{&mtp_v_cache(), true});
        }
        if (packed_targets.empty()) {
            throw std::runtime_error(
                "KVMem packed CPU stage-in has no target tensors");
        }

        uint32_t pages_per_target = 0;
        for (size_t i = begin; i < end; ++i) {
            const KvMemPrefetchCpuRead &read =
                kvmem_prefetch_.cpu_reads[i];
            const KvMemBlock &block = blocks[read.block_id];
            const uint32_t page_size = kv_pages_.page_size;
            const uint32_t first_page =
                block.orig_pos_start / page_size;
            const uint32_t last_page =
                (block.orig_pos_start + block.n_tokens - 1) /
                page_size;
            for (uint32_t lp = first_page; lp <= last_page; ++lp) {
                if (!kv_pages_.logical_page_resident(lp) ||
                    (stage_mtp_pages &&
                     !mtp_kv_pages_.logical_page_resident(lp))) {
                    throw std::runtime_error(
                        "KVMem packed CPU stage-in target page was not "
                        "bulk-allocated");
                }
            }
            pages_per_target += last_page - first_page + 1;
        }
        batch.src_page_indices.resize(
            static_cast<size_t>(pages_per_target) *
            packed_targets.size());
        batch.dst_page_indices.resize(
            batch.src_page_indices.size());
        const uint64_t page_bytes = kvmem_kv_page_bytes();
        for (size_t t = 0; t < packed_targets.size(); ++t) {
            uint32_t page_entry = 0;
            for (size_t i = begin; i < end; ++i) {
                const KvMemPrefetchCpuRead &read =
                    kvmem_prefetch_.cpu_reads[i];
                const KvMemBlock &block = blocks[read.block_id];
                const uint32_t page_size = kv_pages_.page_size;
                const uint32_t first_page =
                    block.orig_pos_start / page_size;
                const uint32_t last_page =
                    (block.orig_pos_start + block.n_tokens - 1) /
                    page_size;
                const uint32_t block_pages =
                    last_page - first_page + 1;
                if (read.batch_offset % page_bytes != 0) {
                    throw std::runtime_error(
                        "KVMem packed CPU stage-in block is not "
                        "page-aligned");
                }
                const uint64_t target_src_base =
                    read.batch_offset / page_bytes +
                    static_cast<uint64_t>(t) * block_pages;
                for (uint32_t p = 0; p < block_pages; ++p) {
                    const size_t index =
                        t * pages_per_target + page_entry;
                    batch.src_page_indices[index] =
                        static_cast<int32_t>(target_src_base + p);
                    const uint32_t logical_page = first_page + p;
                    batch.dst_page_indices[index] =
                        packed_targets[t].mtp
                            ? mtp_kv_pages_.pages[logical_page]
                            : kv_pages_.pages[logical_page];
                    ++page_entry;
                }
            }
        }
        std::vector<DeviceTensor *> target_tensors;
        target_tensors.reserve(packed_targets.size());
        for (const PackedTarget &target : packed_targets) {
            target_tensors.push_back(target.tensor);
        }
        require_status(backend_.copy_packed_pages_from_host_async(
            slab, batch_bytes, target_tensors,
            batch.src_page_indices.data(),
            batch.dst_page_indices.data(),
            pages_per_target, page_bytes));
        require_status(backend_.record_kv_transfer_fence(batch.fence));
        if (perf_trace) {
            kvmem_prefetch_.perf.cpu_h2d_enqueue_ns +=
                kvmem_steady_ns() - enqueue0;
            ++kvmem_prefetch_.perf.cpu_h2d_batches;
        }
        kvmem_prefetch_.next_cpu_read = end;
        kvmem_prefetch_.cpu_batches.push_back(std::move(batch));
    }
}

void QwenExecutor::kvmem_submit_prefetch_nvme_batches() {
    if (!kvmem_prefetch_.bulk_nvme || !kvmem_nvme_tier_) return;
    constexpr size_t kMaxInflightReadBatches = 2;
    const uint64_t read_slab_target_bytes = kvmem_io_slab_bytes();
    while (kvmem_prefetch_.nvme_batches.size() <
               kMaxInflightReadBatches &&
           kvmem_prefetch_.next_nvme_read <
               kvmem_prefetch_.nvme_reads.size()) {
        const size_t begin = kvmem_prefetch_.next_nvme_read;
        size_t end = begin;
        uint64_t batch_bytes = 0;
        while (end < kvmem_prefetch_.nvme_reads.size()) {
            const uint64_t bytes =
                kvmem_prefetch_.nvme_reads[end].bytes;
            if (end > begin &&
                batch_bytes + bytes > read_slab_target_bytes) {
                break;
            }
            kvmem_prefetch_.nvme_reads[end].batch_offset = batch_bytes;
            batch_bytes += bytes;
            ++end;
        }
        KvMemPrefetchNvmeBatch batch;
        batch.read_begin = begin;
        batch.read_end = end;
        for (auto it = kvmem_free_read_slabs_.begin();
             it != kvmem_free_read_slabs_.end(); ++it) {
            if ((*it)->bytes >= batch_bytes) {
                batch.buffer = std::move(*it);
                kvmem_free_read_slabs_.erase(it);
                break;
            }
        }
        if (!batch.buffer) {
            batch.buffer = backend_.host_buffer(
                batch_bytes, "kvmem_stagein_read_slab");
        }
        batch.spans.reserve(end - begin);
        for (size_t i = begin; i < end; ++i) {
            const KvMemPrefetchNvmeRead &read =
                kvmem_prefetch_.nvme_reads[i];
            batch.spans.push_back(
                NvmeIoSpan{read.slot, read.batch_offset, read.bytes});
        }
        NvmeKvTier *tier = kvmem_nvme_tier_.get();
        void *buffer = batch.buffer->data;
        auto spans = batch.spans;
        batch.future = std::async(
            std::launch::async,
            [tier, buffer, batch_bytes, spans = std::move(spans)]() {
                NvmeBatchIoStats stats;
                const uint64_t t0 = kvmem_steady_ns();
                tier->read_spans(spans, buffer, batch_bytes, &stats);
                stats.duration_ns = kvmem_steady_ns() - t0;
                return stats;
            });
        kvmem_prefetch_.next_nvme_read = end;
        ++kvmem_prefetch_.perf.nvme_read_batches;
        kvmem_prefetch_.nvme_batches.push_back(std::move(batch));
    }
}

void QwenExecutor::kvmem_start_prefetch(const KvMemPlan &plan) {
    if (!block_store_) return;
    if (kvmem_prefetch_.active) {
        throw std::runtime_error("KVMem prefetch already active");
    }
    const bool perf_trace = kvmem_perf_trace_flag();
    const bool tm = kvmem_measure_timing_flag();
    const uint64_t t_in0 = tm ? kvmem_steady_ns() : 0;
    kvmem_prefetch_ = KvMemPrefetchState{};
    kvmem_prefetch_.active = true;
    if (perf_trace) {
        kvmem_prefetch_.perf.start_enter_ns = t_in0;
    }
    bool needs_pending_write = false;
    if (!kvmem_pending_writes_.empty()) {
        const auto &blocks = block_store_->blocks();
        for (const KvMemRemap &rm : plan.remaps) {
            if (rm.block_id < blocks.size() &&
                blocks[rm.block_id].tier == KvTier::SSD &&
                blocks[rm.block_id].in_flight) {
                needs_pending_write = true;
                break;
            }
        }
    }
    if (needs_pending_write) {
        const uint64_t wait0 =
            perf_trace ? kvmem_steady_ns() : 0;
        kvmem_reap_pending_writes(/*wait_all=*/true);
        if (perf_trace) {
            kvmem_prefetch_.perf.pending_write_wait_ns +=
                kvmem_steady_ns() - wait0;
        }
    }
    // Local-position MTP stores historical V independently of the prefix
    // builder's current tail, so restore it even when an accepted token has
    // already been registered and its MTP prefix is about to be replayed. The
    // old coverage gate caused CPU-resident V to be silently skipped in this
    // prepare -> replay -> finish interval. Legacy MTP retains the gate: if its
    // prefix path was disabled, the sparse MTP table is intentionally empty and
    // ensuring a far historical logical page would densely allocate every
    // intervening page and exhaust the bounded pool.
    kvmem_prefetch_.stage_mtp_pages =
        kvmem_mtp_tiered_ &&
        (kvmem_mtp_local_positions_ ||
         kvmem_mtp_prefix_covers_registered());
    const bool stage_mtp_pages = kvmem_prefetch_.stage_mtp_pages;
    kvmem_prefetch_.bulk_cpu =
        block_store_->config().packed_rematerialization &&
        kvmem_cpu_tier_ != nullptr;
    kvmem_prefetch_.bulk_nvme =
        block_store_->config().packed_rematerialization &&
        kvmem_nvme_tier_ != nullptr;
    try {
        const auto &blocks = block_store_->blocks();
        require_status(backend_.begin_kv_transfer_to_device());
        for (const KvMemRemap &rm : plan.remaps) {
            const KvMemBlock &block = blocks[rm.block_id];
            if (kvmem_block_pages_resident(block)) {
                block_store_->set_block_tier(rm.block_id, KvTier::GPU);
                continue;
            }
            const uint64_t bytes = kvmem_block_spill_bytes(block);
            const void *stage_src = nullptr;
            uint64_t stage_bytes = bytes;
            if (block.tier == KvTier::CPU) {
                if (!kvmem_cpu_tier_) {
                    throw std::runtime_error("KVMem CPU stage-in has no CPU tier");
                }
                const int32_t slot = kvmem_cpu_tier_->block_slot(rm.block_id);
                if (slot < 0) {
                    throw std::runtime_error("KVMem CPU stage-in missing CPU slot");
                }
                stage_src = kvmem_cpu_slot_data(slot);
                if (!stage_src) {
                    throw std::runtime_error(
                        "KVMem CPU stage-in has no CPU slot backing");
                }
                if (perf_trace) {
                    ++kvmem_prefetch_.perf.cpu_blocks;
                    kvmem_prefetch_.perf.cpu_bytes += bytes;
                }
                if (kvmem_prefetch_.bulk_cpu) {
                    kvmem_prefetch_.cpu_reads.push_back(
                        KvMemPrefetchCpuRead{
                            rm.block_id, bytes, slot, 0});
                    continue;
                }
            } else if (block.tier == KvTier::SSD) {
                if (!kvmem_nvme_tier_) {
                    throw std::runtime_error("KVMem NVMe stage-in has no NVMe tier");
                }
                KvMemPrefetchNvmeRead read;
                read.block_id = rm.block_id;
                read.bytes = bytes;
                read.slot = kvmem_nvme_tier_->block_slot(rm.block_id);
                if (read.slot < 0) {
                    throw std::runtime_error(
                        "KVMem NVMe stage-in missing backing slot");
                }
                if (!kvmem_prefetch_.bulk_nvme) {
                    read.buffer.resize(bytes);
                }
                kvmem_prefetch_.nvme_reads.push_back(std::move(read));
                if (perf_trace) {
                    ++kvmem_prefetch_.perf.nvme_blocks;
                    kvmem_prefetch_.perf.nvme_bytes += bytes;
                }
                if (kvmem_tier_trace_enabled()) {
                    std::fprintf(stderr,
                                 "[kvmem-tier] stage_in_async_read block=%u from=nvme bytes=%llu\n",
                                 rm.block_id,
                                 static_cast<unsigned long long>(bytes));
                }
                continue;
            } else {
                throw std::runtime_error(
                    "KVMem stage-in requested a non-resident block with no backing tier");
            }
            const uint64_t t_cpu_enqueue0 =
                perf_trace ? kvmem_steady_ns() : 0;
            const uint32_t page_size = kv_pages_.page_size;
            const uint32_t first_page = block.orig_pos_start / page_size;
            const uint32_t last_page =
                (block.orig_pos_start + block.n_tokens - 1) / page_size;
            for (uint32_t lp = first_page; lp <= last_page; ++lp) {
                (void)kv_pages_.ensure_logical_page_resident(backend_, lp);
                if (stage_mtp_pages) {
                    (void)mtp_kv_pages_.ensure_logical_page_resident(backend_, lp);
                }
            }
            kvmem_copy_block_from_host(block, stage_src, stage_bytes,
                                       stage_mtp_pages);
            if (perf_trace) {
                kvmem_prefetch_.perf.cpu_h2d_enqueue_ns +=
                    kvmem_steady_ns() - t_cpu_enqueue0;
            }
            kvmem_prefetch_.queued_h2d = true;
            kvmem_prefetch_.blocks.push_back(KvMemPrefetchBlock{rm.block_id,
                                                                block.tier});
            if (kvmem_tier_trace_enabled()) {
                std::fprintf(stderr,
                             "[kvmem-tier] stage_in block=%u from=%s bytes=%llu\n",
                             rm.block_id,
                             block.tier == KvTier::CPU ? "cpu" : "nvme",
                             static_cast<unsigned long long>(bytes));
            }
        }
        if ((kvmem_prefetch_.bulk_cpu &&
             !kvmem_prefetch_.cpu_reads.empty()) ||
            (kvmem_prefetch_.bulk_nvme &&
             !kvmem_prefetch_.nvme_reads.empty())) {
            // Publish all incoming page mappings in one compact upload. This
            // is required for the CPU overlap (assembly reads an immutable
            // page vector), and also removes thousands of redundant scalar
            // page-table H2D uploads from ordinary opt_3 stage-in.
            std::vector<std::pair<uint32_t, uint32_t>> page_ranges;
            page_ranges.reserve(
                kvmem_prefetch_.cpu_reads.size() +
                kvmem_prefetch_.nvme_reads.size());
            auto add_range = [&](uint32_t block_id) {
                const KvMemBlock &block = blocks[block_id];
                const uint32_t page_size = kv_pages_.page_size;
                const uint32_t first_page =
                    block.orig_pos_start / page_size;
                const uint32_t last_page =
                    (block.orig_pos_start + block.n_tokens - 1) /
                    page_size;
                page_ranges.emplace_back(
                    first_page, last_page - first_page + 1);
            };
            if (kvmem_prefetch_.bulk_cpu) {
                for (const KvMemPrefetchCpuRead &read :
                     kvmem_prefetch_.cpu_reads) {
                    add_range(read.block_id);
                }
            }
            if (kvmem_prefetch_.bulk_nvme) {
                for (const KvMemPrefetchNvmeRead &read :
                     kvmem_prefetch_.nvme_reads) {
                    add_range(read.block_id);
                }
            }
            kv_pages_.ensure_logical_page_ranges_resident(
                backend_, page_ranges);
            if (stage_mtp_pages) {
                mtp_kv_pages_.ensure_logical_page_ranges_resident(
                    backend_, page_ranges);
            }
        }
        // In the unified CPU-only pipeline, do not gather the first two V
        // slabs on the request thread. Starting them here left a measured
        // ~26 ms/reselection serial prefix before raw-K assembly could begin.
        // kvmem_finish_prefetch() runs on the dedicated stage-in caller and
        // submits the same bounded two-slab queue there instead.
        if (!kvmem_prefetch_.cpu_reads.empty() &&
            !kvmem_stagein_assembly_overlap_enabled_) {
            kvmem_submit_prefetch_cpu_batches();
        }
        if (!kvmem_prefetch_.nvme_reads.empty()) {
            if (kvmem_prefetch_.bulk_nvme) {
                kvmem_submit_prefetch_nvme_batches();
            } else {
                NvmeKvTier *tier = kvmem_nvme_tier_.get();
                std::vector<KvMemPrefetchNvmeRead> *reads =
                    &kvmem_prefetch_.nvme_reads;
                uint64_t *read_ns = &kvmem_prefetch_.perf.nvme_read_ns;
                kvmem_prefetch_.nvme_future = std::async(
                    std::launch::async,
                    [tier, reads, read_ns, perf_trace]() {
                        const uint64_t t0 =
                            perf_trace ? kvmem_steady_ns() : 0;
                        for (KvMemPrefetchNvmeRead &read : *reads) {
                            tier->read_block(
                                read.block_id, read.buffer.data(),
                                read.bytes);
                        }
                        if (perf_trace) {
                            *read_ns = kvmem_steady_ns() - t0;
                        }
                    });
            }
        }
    } catch (...) {
        kvmem_prefetch_ = KvMemPrefetchState{};
        throw;
    }
    if (perf_trace) {
        kvmem_prefetch_.perf.start_exit_ns = kvmem_steady_ns();
    }
    if (tm) {
        kvmem_timing_totals().stage_in_ns.fetch_add(
            kvmem_steady_ns() - t_in0, std::memory_order_relaxed);
    }
}

void QwenExecutor::kvmem_finish_prefetch() {
    if (!kvmem_prefetch_.active) return;
    const bool perf_trace = kvmem_perf_trace_flag();
    const bool tm = kvmem_measure_timing_flag();
    const uint64_t t_in0 = tm ? kvmem_steady_ns() : 0;
    if (perf_trace) {
        kvmem_prefetch_.perf.finish_enter_ns = t_in0;
    }
    uint32_t staged = 0;
    const bool stage_mtp_pages = kvmem_prefetch_.stage_mtp_pages;
    try {
        if (kvmem_prefetch_.bulk_cpu) {
            if (kvmem_prefetch_.cpu_batches.empty() &&
                kvmem_prefetch_.next_cpu_read <
                    kvmem_prefetch_.cpu_reads.size()) {
                kvmem_submit_prefetch_cpu_batches();
            }
            while (!kvmem_prefetch_.cpu_batches.empty()) {
                KvMemPrefetchCpuBatch batch =
                    std::move(kvmem_prefetch_.cpu_batches.front());
                kvmem_prefetch_.cpu_batches.pop_front();
                const uint64_t wait0 =
                    perf_trace ? kvmem_steady_ns() : 0;
                require_status(
                    backend_.wait_kv_transfer_fence(batch.fence));
                if (perf_trace) {
                    kvmem_prefetch_.perf.cpu_h2d_wait_ns +=
                        kvmem_steady_ns() - wait0;
                }
                for (size_t i = batch.read_begin;
                     i < batch.read_end; ++i) {
                    const KvMemPrefetchCpuRead &read =
                        kvmem_prefetch_.cpu_reads[i];
                    kvmem_prefetch_.blocks.push_back(
                        KvMemPrefetchBlock{
                            read.block_id, KvTier::CPU});
                    if (kvmem_tier_trace_enabled()) {
                        std::fprintf(
                            stderr,
                            "[kvmem-tier] stage_in block=%u "
                            "from=cpu bytes=%llu mode=batched\n",
                            read.block_id,
                            static_cast<unsigned long long>(
                                read.bytes));
                    }
                }
                if (batch.buffer &&
                    kvmem_free_read_slabs_.size() <
                        kvmem_read_slab_count_) {
                    kvmem_free_read_slabs_.push_back(
                        std::move(batch.buffer));
                }
                // H2D for the next slab is already in flight. Gathering the
                // newly submitted slab here overlaps host memcpy with it.
                kvmem_submit_prefetch_cpu_batches();
            }
        }
        if (kvmem_prefetch_.bulk_nvme) {
            while (!kvmem_prefetch_.nvme_batches.empty()) {
                KvMemPrefetchNvmeBatch batch =
                    std::move(kvmem_prefetch_.nvme_batches.front());
                kvmem_prefetch_.nvme_batches.pop_front();
                const uint64_t t_wait0 =
                    perf_trace ? kvmem_steady_ns() : 0;
                NvmeBatchIoStats io = batch.future.get();
                if (perf_trace) {
                    kvmem_prefetch_.perf.nvme_wait_ns +=
                        kvmem_steady_ns() - t_wait0;
                    kvmem_prefetch_.perf.nvme_read_ns += io.duration_ns;
                    kvmem_prefetch_.perf.nvme_read_syscalls += io.syscalls;
                }
                const size_t read_begin = batch.read_begin;
                const size_t read_end = batch.read_end;
                HostBuffer *host = batch.buffer.get();
                const uint64_t t_nvme_h2d0 =
                    perf_trace ? kvmem_steady_ns() : 0;
                const auto &blocks = block_store_->blocks();
                for (size_t i = read_begin; i < read_end; ++i) {
                    KvMemPrefetchNvmeRead &read =
                        kvmem_prefetch_.nvme_reads[i];
                    const KvMemBlock &block = blocks[read.block_id];
                    const uint32_t page_size = kv_pages_.page_size;
                    const uint32_t first_page =
                        block.orig_pos_start / page_size;
                    const uint32_t last_page =
                        (block.orig_pos_start + block.n_tokens - 1) /
                        page_size;
                    for (uint32_t lp = first_page; lp <= last_page; ++lp) {
                        if (!kv_pages_.logical_page_resident(lp) ||
                            (stage_mtp_pages &&
                             !mtp_kv_pages_.logical_page_resident(lp))) {
                            throw std::runtime_error(
                                "KVMem packed NVMe stage-in target page was "
                                "not bulk-allocated");
                        }
                    }
                    const uint8_t *src =
                        static_cast<const uint8_t *>(host->data) +
                        read.batch_offset;
                    kvmem_copy_block_from_host(
                        block, src, read.bytes, stage_mtp_pages);
                    kvmem_prefetch_.queued_h2d = true;
                    kvmem_prefetch_.blocks.push_back(
                        KvMemPrefetchBlock{read.block_id, KvTier::SSD});
                }
                if (perf_trace) {
                    kvmem_prefetch_.perf.nvme_h2d_enqueue_ns +=
                        kvmem_steady_ns() - t_nvme_h2d0;
                }
                if (kvmem_prefetch_.queued_h2d) {
                    const uint64_t t_h2d_wait0 =
                        perf_trace ? kvmem_steady_ns() : 0;
                    require_status(backend_.wait_kv_transfer());
                    if (perf_trace) {
                        kvmem_prefetch_.perf.h2d_wait_ns +=
                            kvmem_steady_ns() - t_h2d_wait0;
                    }
                    kvmem_prefetch_.queued_h2d = false;
                }
                if (batch.buffer &&
                    kvmem_free_read_slabs_.size() <
                        kvmem_read_slab_count_) {
                    kvmem_free_read_slabs_.push_back(
                        std::move(batch.buffer));
                }
                // The second read submitted at start remained outstanding
                // during this slab's H2D. Refill the just-freed queue slot only
                // after returning its pinned slab, so later batches reuse the
                // same bounded buffers instead of paying cudaHostAlloc in the
                // stage-in critical path.
                kvmem_submit_prefetch_nvme_batches();
            }
        } else if (kvmem_prefetch_.nvme_future.valid()) {
            const uint64_t t_wait0 =
                perf_trace ? kvmem_steady_ns() : 0;
            kvmem_prefetch_.nvme_future.get();
            if (perf_trace) {
                kvmem_prefetch_.perf.nvme_wait_ns +=
                    kvmem_steady_ns() - t_wait0;
            }
        }
        if (!kvmem_prefetch_.bulk_nvme &&
            !kvmem_prefetch_.nvme_reads.empty()) {
            const uint64_t t_nvme_h2d0 =
                perf_trace ? kvmem_steady_ns() : 0;
            const auto &blocks = block_store_->blocks();
            for (KvMemPrefetchNvmeRead &read : kvmem_prefetch_.nvme_reads) {
                const KvMemBlock &block = blocks[read.block_id];
                const uint32_t page_size = kv_pages_.page_size;
                const uint32_t first_page = block.orig_pos_start / page_size;
                const uint32_t last_page =
                    (block.orig_pos_start + block.n_tokens - 1) / page_size;
                for (uint32_t lp = first_page; lp <= last_page; ++lp) {
                    (void)kv_pages_.ensure_logical_page_resident(backend_, lp);
                    if (stage_mtp_pages) {
                        (void)mtp_kv_pages_.ensure_logical_page_resident(backend_, lp);
                    }
                }
                kvmem_copy_block_from_host(block, read.buffer.data(),
                                           read.bytes, stage_mtp_pages);
                kvmem_prefetch_.queued_h2d = true;
                kvmem_prefetch_.blocks.push_back(
                    KvMemPrefetchBlock{read.block_id, KvTier::SSD});
                if (kvmem_tier_trace_enabled()) {
                    std::fprintf(stderr,
                                 "[kvmem-tier] stage_in block=%u from=nvme bytes=%llu\n",
                                 read.block_id,
                                 static_cast<unsigned long long>(read.bytes));
                }
            }
            if (perf_trace) {
                kvmem_prefetch_.perf.nvme_h2d_enqueue_ns +=
                    kvmem_steady_ns() - t_nvme_h2d0;
            }
        }
        if (kvmem_prefetch_.queued_h2d) {
            const uint64_t t_h2d_wait0 =
                perf_trace ? kvmem_steady_ns() : 0;
            require_status(backend_.wait_kv_transfer());
            if (perf_trace) {
                kvmem_prefetch_.perf.h2d_wait_ns +=
                    kvmem_steady_ns() - t_h2d_wait0;
            }
        }
        staged = static_cast<uint32_t>(kvmem_prefetch_.blocks.size());
        const bool inclusive =
            block_store_->config().optimization_level >=
                KvMemOptimizationLevel::Opt2 &&
            (kvmem_nvme_tier_ != nullptr ||
             kvmem_inclusive_cpu_backing_);
        for (const KvMemPrefetchBlock &pb : kvmem_prefetch_.blocks) {
            if (pb.from == KvTier::CPU && kvmem_cpu_tier_) {
                if (!inclusive) {
                    const int32_t slot =
                        kvmem_cpu_tier_->block_slot(pb.block_id);
                    kvmem_cpu_tier_->release_block(pb.block_id);
                    kvmem_release_cpu_slot(slot);
                    block_store_->set_block_cpu_copy(
                        pb.block_id, -1);
                }
            } else if (pb.from == KvTier::SSD && kvmem_nvme_tier_) {
                if (!inclusive) {
                    kvmem_nvme_tier_->release_block(pb.block_id);
                }
            }
            block_store_->set_block_tier(pb.block_id, KvTier::GPU);
        }
    } catch (...) {
        kvmem_prefetch_ = KvMemPrefetchState{};
        throw;
    }
    if (perf_trace) {
        kvmem_prefetch_.perf.finish_exit_ns = kvmem_steady_ns();
        kvmem_last_prefetch_perf_ = kvmem_prefetch_.perf;
    }
    kvmem_prefetch_ = KvMemPrefetchState{};
    if (tm) {
        KvMemTimingTotals &t = kvmem_timing_totals();
        t.stage_in_ns.fetch_add(kvmem_steady_ns() - t_in0,
                                std::memory_order_relaxed);
        t.stage_in_calls.fetch_add(1, std::memory_order_relaxed);
        t.stage_in_blocks.fetch_add(staged, std::memory_order_relaxed);
    }
}

void QwenExecutor::kvmem_reap_pending_writes(bool wait_all) {
    using namespace std::chrono_literals;
    while (!kvmem_pending_writes_.empty()) {
        KvMemPendingWriteBatch &front = kvmem_pending_writes_.front();
        if (!wait_all &&
            front.future.wait_for(0ms) != std::future_status::ready) {
            break;
        }
        NvmeBatchIoStats stats;
        try {
            stats = front.future.get();
        } catch (...) {
            if (block_store_) {
                for (uint32_t id : front.block_ids) {
                    block_store_->set_block_io_in_flight(id, false);
                }
            }
            if (kvmem_cpu_tier_) {
                for (const auto &copy : front.cpu_copies) {
                    if (kvmem_cpu_tier_->block_slot(copy.block_id) ==
                        copy.slot) {
                        kvmem_cpu_tier_->release_block(copy.block_id);
                        kvmem_release_cpu_slot(copy.slot);
                        if (block_store_) {
                            block_store_->set_block_cpu_copy(
                                copy.block_id, -1);
                        }
                    }
                }
            }
            kvmem_pending_writes_.pop_front();
            throw;
        }
        if (block_store_ && kvmem_nvme_tier_) {
            for (size_t i = 0; i < front.block_ids.size(); ++i) {
                const uint32_t id = front.block_ids[i];
                if (id >= block_store_->block_count() ||
                    i >= front.spans.size()) {
                    continue;
                }
                const int32_t slot = front.spans[i].slot;
                if (kvmem_nvme_tier_->block_slot(id) == slot) {
                    block_store_->set_block_ssd_backing(id, slot, true);
                }
                block_store_->set_block_io_in_flight(id, false);
            }
        }
        if (block_store_ && kvmem_cpu_tier_) {
            for (const auto &copy : front.cpu_copies) {
                if (copy.block_id >= block_store_->block_count()) {
                    continue;
                }
                if (kvmem_cpu_tier_->block_slot(copy.block_id) ==
                    copy.slot) {
                    const KvMemBlock &block =
                        block_store_->blocks()[copy.block_id];
                    if (kvmem_block_pages_resident(block)) {
                        block_store_->set_block_cpu_copy(
                            copy.block_id, copy.slot);
                    } else {
                        block_store_->set_block_tier(
                            copy.block_id, KvTier::CPU,
                            copy.slot, block.nvme_slot);
                    }
                }
                // SSD write-back clears this above when its durable write
                // completes. CPU-only proactive write-back has no SSD loop, so
                // publish the clean CPU authority here.
                if (!kvmem_nvme_tier_) {
                    block_store_->set_block_io_in_flight(
                        copy.block_id, false);
                }
            }
        }
        kvmem_async_write_completed_bytes_ += stats.bytes;
        kvmem_async_write_completed_syscalls_ += stats.syscalls;
        kvmem_async_write_completed_ns_ += stats.duration_ns;
        std::shared_ptr<std::vector<uint8_t>> completed_buffer =
            std::move(front.buffer);
        std::shared_ptr<HostBuffer> completed_pinned =
            std::move(front.pinned_buffer);
        const uint64_t completed_bytes = front.buffer_bytes;
        const bool completed_proactive = front.proactive;
        const bool completed_cpu_only = front.spans.empty();
        const uint32_t completed_blocks =
            static_cast<uint32_t>(front.block_ids.size());
        kvmem_pending_writes_.pop_front();
        if (completed_buffer &&
            kvmem_free_write_slabs_.size() <
                kvmem_write_queue_depth()) {
            completed_buffer->clear();
            kvmem_free_write_slabs_.push_back(
                std::move(completed_buffer));
        }
        if (completed_pinned &&
            kvmem_free_writeback_slabs_.size() <
                kvmem_writeback_slab_count_) {
            kvmem_free_writeback_slabs_.push_back(
                std::move(completed_pinned));
        }
        if (kvmem_perf_trace_flag() && completed_bytes > 0) {
            std::fprintf(
                stderr,
                "[kvmem-writeback] event=%s_complete blocks=%u "
                "bytes=%llu duration_ms=%.3f pending_ssd=%zu "
                "source=%s cpu_copy_blocks=%u "
                "cpu_copy_mib=%.3f cpu_copy_ms=%.3f\n",
                completed_cpu_only ? "cpu" : "ssd",
                completed_blocks,
                static_cast<unsigned long long>(completed_bytes),
                stats.duration_ns / 1.0e6,
                kvmem_pending_writes_.size(),
                completed_proactive ? "prefill" : "pressure",
                stats.cpu_copy_blocks,
                stats.cpu_copy_bytes / (1024.0 * 1024.0),
                stats.cpu_copy_ns / 1.0e6);
        }
    }
}

void QwenExecutor::kvmem_submit_write_batch(
        std::shared_ptr<std::vector<uint8_t>> buffer,
        std::vector<NvmeIoSpan> spans,
        std::vector<uint32_t> block_ids) {
    if (!buffer || buffer->empty() || spans.empty()) return;
    if (!kvmem_nvme_tier_) {
        throw std::runtime_error(
            "KVMem async write submission has no NVMe tier");
    }
    // QD=8 is the portable default because it saturates the profiled device
    // without a large queue. Deployments can tune the bounded queue through
    // QW3_KVMEM_WRITE_QUEUE_DEPTH after running profile_kvmem_storage.py.
    const size_t max_inflight_batches = kvmem_write_queue_depth();
    const uint64_t wait0 = kvmem_steady_ns();
    if (kvmem_pending_writes_.size() >= max_inflight_batches) {
        kvmem_pending_writes_.front().future.wait();
        kvmem_reap_pending_writes(/*wait_all=*/false);
    }
    kvmem_last_stage_out_perf_.async_backpressure_ns +=
        kvmem_steady_ns() - wait0;

    NvmeKvTier *tier = kvmem_nvme_tier_.get();
    KvMemPendingWriteBatch pending;
    pending.buffer = std::move(buffer);
    pending.buffer_bytes = pending.buffer->size();
    pending.spans = std::move(spans);
    pending.block_ids = std::move(block_ids);
    pending.submit_ns = kvmem_steady_ns();
    auto worker_buffer = pending.buffer;
    auto worker_spans = pending.spans;
    pending.future = std::async(
        std::launch::async,
        [tier, worker_buffer, worker_spans = std::move(worker_spans)]() {
            NvmeBatchIoStats stats;
            const uint64_t t0 = kvmem_steady_ns();
            tier->write_spans(
                worker_spans, worker_buffer->data(),
                static_cast<uint64_t>(worker_buffer->size()), &stats);
            stats.duration_ns = kvmem_steady_ns() - t0;
            return stats;
        });
    kvmem_last_stage_out_perf_.async_submitted_bytes +=
        pending.buffer->size();
    ++kvmem_last_stage_out_perf_.async_batches;
    kvmem_pending_writes_.push_back(std::move(pending));
}

void QwenExecutor::kvmem_submit_pinned_write_batch(
        std::shared_ptr<HostBuffer> buffer, uint64_t bytes,
        std::vector<NvmeIoSpan> spans,
        std::vector<uint32_t> block_ids,
        std::vector<KvMemPendingWriteBatch::CpuCopy> cpu_copies) {
    if (!buffer || !buffer->data || bytes == 0 || spans.empty()) return;
    if (bytes > buffer->bytes) {
        throw std::runtime_error(
            "KVMem proactive write-back exceeds its pinned slab");
    }
    if (!kvmem_nvme_tier_) {
        throw std::runtime_error(
            "KVMem proactive write submission has no NVMe tier");
    }
    const size_t max_inflight_batches =
        kvmem_write_queue_depth();
    const uint64_t wait0 = kvmem_steady_ns();
    if (kvmem_pending_writes_.size() >= max_inflight_batches) {
        kvmem_pending_writes_.front().future.wait();
        kvmem_reap_pending_writes(/*wait_all=*/false);
    }
    kvmem_last_stage_out_perf_.async_backpressure_ns +=
        kvmem_steady_ns() - wait0;

    NvmeKvTier *tier = kvmem_nvme_tier_.get();
    KvMemPendingWriteBatch pending;
    pending.pinned_buffer = std::move(buffer);
    pending.buffer_bytes = bytes;
    pending.proactive = true;
    pending.spans = std::move(spans);
    pending.block_ids = std::move(block_ids);
    pending.cpu_copies = std::move(cpu_copies);
    pending.submit_ns = kvmem_steady_ns();
    auto worker_buffer = pending.pinned_buffer;
    auto worker_spans = pending.spans;
    auto worker_cpu_copies = pending.cpu_copies;
    pending.future = std::async(
        std::launch::async,
        [tier, worker_buffer, bytes,
         worker_spans = std::move(worker_spans),
         worker_cpu_copies = std::move(worker_cpu_copies)]() {
            NvmeBatchIoStats stats;
            const uint64_t cpu0 = kvmem_steady_ns();
            uint64_t cpu_copy_bytes = 0;
            for (const auto &copy : worker_cpu_copies) {
                std::memcpy(
                    copy.dst,
                    static_cast<const uint8_t *>(
                        worker_buffer->data) +
                        copy.buffer_offset,
                    static_cast<size_t>(copy.bytes));
                cpu_copy_bytes += copy.bytes;
            }
            const uint64_t cpu_copy_ns =
                kvmem_steady_ns() - cpu0;
            const uint64_t io0 = kvmem_steady_ns();
            tier->write_spans(
                worker_spans, worker_buffer->data,
                bytes, &stats);
            // write_spans initializes its output structure, so attach the
            // independent CPU-cache measurements after positional I/O.
            stats.cpu_copy_bytes = cpu_copy_bytes;
            stats.cpu_copy_blocks =
                static_cast<uint32_t>(
                    worker_cpu_copies.size());
            stats.cpu_copy_ns = cpu_copy_ns;
            stats.duration_ns = kvmem_steady_ns() - io0;
            return stats;
        });
    kvmem_last_stage_out_perf_.async_submitted_bytes += bytes;
    ++kvmem_last_stage_out_perf_.async_batches;
    kvmem_pending_writes_.push_back(std::move(pending));
}

void QwenExecutor::kvmem_submit_pinned_cpu_copy_batch(
        std::shared_ptr<HostBuffer> buffer, uint64_t bytes,
        std::vector<uint32_t> block_ids,
        std::vector<KvMemPendingWriteBatch::CpuCopy> cpu_copies) {
    if (!buffer || !buffer->data || bytes == 0 ||
        block_ids.empty() || cpu_copies.empty()) {
        return;
    }
    if (bytes > buffer->bytes) {
        throw std::runtime_error(
            "KVMem proactive CPU write-back exceeds its pinned slab");
    }
    if (!kvmem_cpu_tier_ || kvmem_nvme_tier_) {
        throw std::runtime_error(
            "KVMem proactive CPU copy requires a CPU-only tier");
    }

    // Bound both worker count and pinned-slab ownership. In the normal
    // double-buffered pipeline the oldest scatter finishes during the next
    // prefill chunk, so this wait is normally zero.
    const size_t max_inflight_batches =
        std::max<size_t>(1, kvmem_writeback_slab_count_);
    const uint64_t wait0 = kvmem_steady_ns();
    if (kvmem_pending_writes_.size() >= max_inflight_batches) {
        kvmem_pending_writes_.front().future.wait();
        kvmem_reap_pending_writes(/*wait_all=*/false);
    }
    kvmem_last_stage_out_perf_.async_backpressure_ns +=
        kvmem_steady_ns() - wait0;

    KvMemPendingWriteBatch pending;
    pending.pinned_buffer = std::move(buffer);
    pending.buffer_bytes = bytes;
    pending.proactive = true;
    pending.block_ids = std::move(block_ids);
    pending.cpu_copies = std::move(cpu_copies);
    pending.submit_ns = kvmem_steady_ns();
    auto worker_buffer = pending.pinned_buffer;
    auto worker_cpu_copies = pending.cpu_copies;
    pending.future = std::async(
        std::launch::async,
        [worker_buffer,
         worker_cpu_copies = std::move(worker_cpu_copies)]() {
            NvmeBatchIoStats stats;
            const uint64_t t0 = kvmem_steady_ns();
            uint64_t copied = 0;
            for (const auto &copy : worker_cpu_copies) {
                std::memcpy(
                    copy.dst,
                    static_cast<const uint8_t *>(
                        worker_buffer->data) +
                        copy.buffer_offset,
                    static_cast<size_t>(copy.bytes));
                copied += copy.bytes;
            }
            stats.bytes = copied;
            stats.cpu_copy_bytes = copied;
            stats.cpu_copy_blocks =
                static_cast<uint32_t>(worker_cpu_copies.size());
            stats.cpu_copy_ns = kvmem_steady_ns() - t0;
            stats.duration_ns = stats.cpu_copy_ns;
            return stats;
        });
    kvmem_last_stage_out_perf_.async_submitted_bytes += bytes;
    ++kvmem_last_stage_out_perf_.async_batches;
    kvmem_pending_writes_.push_back(std::move(pending));
}

void QwenExecutor::kvmem_finish_proactive_d2h(bool wait_all) {
    // Publish any already-completed CPU cache copies before admission makes an
    // eviction decision for the next chunk.
    kvmem_reap_pending_writes(/*wait_all=*/false);
    while (!kvmem_proactive_d2h_batches_.empty()) {
        if (!wait_all) break;
        KvMemProactiveD2hBatch batch =
            std::move(kvmem_proactive_d2h_batches_.front());
        kvmem_proactive_d2h_batches_.pop_front();
        const uint64_t wait0 = kvmem_steady_ns();
        const double overlap_window_ms =
            (wait0 - batch.submit_ns) / 1.0e6;
        require_status(
            backend_.wait_kv_transfer_fence(batch.fence));
        const uint64_t wait_ns =
            kvmem_steady_ns() - wait0;
        kvmem_writeback_d2h_wait_ns_ += wait_ns;

        if (!block_store_) {
            throw std::runtime_error(
                "KVMem proactive D2H completed after its block store reset");
        }
        if (!kvmem_nvme_tier_) {
            if (!kvmem_cpu_tier_ ||
                !kvmem_inclusive_cpu_backing_ ||
                batch.cpu_copies.size() != batch.block_ids.size()) {
                throw std::runtime_error(
                    "KVMem proactive CPU D2H lost its inclusive backing");
            }
            const uint32_t block_count =
                static_cast<uint32_t>(batch.block_ids.size());
            const uint64_t bytes = batch.bytes;
            const uint32_t completed_pos = batch.completed_pos;
            kvmem_submit_pinned_cpu_copy_batch(
                std::move(batch.buffer), bytes,
                std::move(batch.block_ids),
                std::move(batch.cpu_copies));
            if (kvmem_perf_trace_flag()) {
                std::fprintf(
                    stderr,
                    "[kvmem-writeback] event=cpu_submit "
                    "completed_pos=%u blocks=%u bytes=%llu "
                    "d2h_wait_ms=%.3f overlap_window_ms=%.3f "
                    "pending_cpu=%zu\n",
                    completed_pos, block_count,
                    static_cast<unsigned long long>(bytes),
                    wait_ns / 1.0e6, overlap_window_ms,
                    kvmem_pending_writes_.size());
            }
            continue;
        }
        uint64_t cpu_copy_bytes = 0;
        std::vector<KvMemPendingWriteBatch::CpuCopy>
            cpu_copies;
        for (size_t i = 0; i < batch.block_ids.size(); ++i) {
            const uint32_t id = batch.block_ids[i];
            if (id >= block_store_->block_count() ||
                i >= batch.spans.size()) {
                continue;
            }
            const int32_t slot = batch.spans[i].slot;
            if (kvmem_nvme_tier_->block_slot(id) != slot) {
                throw std::runtime_error(
                    "KVMem proactive write-back lost its reserved SSD slot");
            }
            // Only now is it safe for pressure selection to release the GPU
            // page: the pinned slab owns a complete immutable copy.
            block_store_->set_block_ssd_backing(id, slot, false);
            block_store_->set_block_io_in_flight(id, true);

            // Preserve the existing inclusive CPU-cache behavior. Without
            // this admission, proactive persistence would make every later
            // miss come from SSD even when the configured CPU tier has room.
            // The copy is deliberately completed at this ownership boundary:
            // raw-K growth in the next chunk may evict CPU slots, so a worker
            // thread must not retain an untracked pointer into a sparse slab.
            const bool admit_cpu_copy =
                block_store_ &&
                (block_store_->config().legacy_optimization_profile
                     ? env_flag_enabled(
                           "QW3_KVMEM_PREFILL_CPU_ADMIT", true)
                     : true);
            if (!kvmem_cpu_tier_ || !admit_cpu_copy) {
                continue;
            }
            const uint64_t bytes = batch.spans[i].bytes;
            PinnedSlotPlacement placement;
            if (kvmem_sparse_cpu_tier_) {
                if (kvmem_cpu_budget_has(bytes)) {
                    placement = kvmem_cpu_tier_->place_block(id);
                    if (placement.slot >= 0 &&
                        !kvmem_reserve_cpu_slot(placement.slot)) {
                        kvmem_cpu_tier_->release_block(id);
                        placement = PinnedSlotPlacement{};
                    }
                } else {
                    // Do not recycle a slot whose background CPU copy still
                    // owns its destination pointer.
                    if (!kvmem_pending_writes_.empty()) {
                        kvmem_reap_pending_writes(
                            /*wait_all=*/true);
                    }
                    placement =
                        kvmem_cpu_tier_->place_block_replacing(id);
                }
            } else {
                if (kvmem_cpu_tier_->free_slots() == 0 &&
                    !kvmem_pending_writes_.empty()) {
                    kvmem_reap_pending_writes(
                        /*wait_all=*/true);
                }
                placement =
                    kvmem_cpu_tier_->place_block_evicting(id);
            }
            if (placement.slot < 0) continue;
            if (placement.evicted_block >= 0) {
                const uint32_t victim =
                    static_cast<uint32_t>(
                        placement.evicted_block);
                if (victim >= block_store_->block_count()) {
                    throw std::runtime_error(
                        "KVMem proactive CPU admission evicted an "
                        "unknown block");
                }
                const KvMemBlock &victim_block =
                    block_store_->blocks()[victim];
                const bool victim_backed =
                    (victim_block.ssd_clean ||
                     victim_block.in_flight) &&
                    victim_block.nvme_slot >= 0 &&
                    kvmem_nvme_tier_->block_slot(victim) ==
                        victim_block.nvme_slot;
                if (!victim_backed) {
                    throw std::runtime_error(
                        "KVMem proactive CPU admission would evict "
                        "the last durable copy of a block");
                }
                block_store_->set_block_cpu_copy(victim, -1);
                if (!kvmem_block_pages_resident(victim_block)) {
                    block_store_->set_block_tier(
                        victim, KvTier::SSD, -1,
                        victim_block.nvme_slot);
                }
            }
            uint8_t *dst =
                kvmem_cpu_slot_data(placement.slot);
            if (!dst) {
                throw std::runtime_error(
                    "KVMem proactive CPU admission lost its slot");
            }
            cpu_copy_bytes += bytes;
            cpu_copies.push_back(
                KvMemPendingWriteBatch::CpuCopy{
                    id, placement.slot, dst,
                    batch.spans[i].buffer_offset, bytes});
        }
        const uint32_t block_count =
            static_cast<uint32_t>(batch.block_ids.size());
        const uint64_t bytes = batch.bytes;
        const uint32_t completed_pos = batch.completed_pos;
        const size_t cpu_copy_blocks = cpu_copies.size();
        kvmem_submit_pinned_write_batch(
            std::move(batch.buffer), bytes,
            std::move(batch.spans),
            std::move(batch.block_ids),
            std::move(cpu_copies));
        if (kvmem_perf_trace_flag()) {
            std::fprintf(
                stderr,
                "[kvmem-writeback] event=ssd_submit completed_pos=%u "
                "blocks=%u bytes=%llu d2h_wait_ms=%.3f "
                "overlap_window_ms=%.3f cpu_admit_blocks=%zu "
                "cpu_admit_mib=%.3f "
                "pending_ssd=%zu\n",
                completed_pos, block_count,
                static_cast<unsigned long long>(bytes),
                wait_ns / 1.0e6,
                overlap_window_ms,
                cpu_copy_blocks,
                cpu_copy_bytes / (1024.0 * 1024.0),
                kvmem_pending_writes_.size());
        }
    }
}

void QwenExecutor::kvmem_prefill_writeback(uint32_t completed_pos) {
    if (!kvmem_prefill_writeback_enabled_ || !block_store_ ||
        (!kvmem_nvme_tier_ && !kvmem_cpu_tier_) ||
        kvmem_query_replay_active_) {
        return;
    }
    // Batch N's D2H was allowed to overlap the just-finished prefill chunk.
    // Finish it now, submit its independent SSD write, then reuse returned
    // slabs for newly-completed blocks from this chunk.
    kvmem_finish_proactive_d2h(/*wait_all=*/true);
    kvmem_reap_pending_writes(/*wait_all=*/false);

    const KvMemStoreConfig &cfg = block_store_->config();
    const auto &blocks = block_store_->blocks();
    std::vector<uint32_t> candidates;
    while (kvmem_writeback_next_block_ < blocks.size()) {
        const uint32_t id = kvmem_writeback_next_block_;
        const KvMemBlock &block = blocks[id];
        const uint64_t block_end =
            static_cast<uint64_t>(block.orig_pos_start) +
            block.n_tokens;
        if (block.n_tokens < cfg.block_tokens ||
            block_end > completed_pos) {
            break;
        }
        ++kvmem_writeback_next_block_;
        if (!kvmem_block_pages_resident(block) ||
            block.ssd_clean || block.in_flight ||
            (!kvmem_nvme_tier_ && block.cpu_slot >= 0 &&
             kvmem_cpu_tier_->block_slot(id) == block.cpu_slot &&
             kvmem_cpu_slot_data(block.cpu_slot))) {
            continue;
        }
        if (kvmem_mtp_tiered_ &&
            !kvmem_block_mtp_pages_resident(block)) {
            throw std::runtime_error(
                "KVMem prefill write-back reached a block before its "
                "MTP V page was durable");
        }
        candidates.push_back(id);
    }
    if (candidates.empty()) return;

    struct PackedTarget {
        DeviceTensor *tensor = nullptr;
        bool mtp = false;
    };
    std::vector<PackedTarget> packed_targets;
    packed_targets.reserve(kvmem_raw_layers_.size() + 1);
    const QwenConfig &model_cfg = model_.config();
    for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
        if (!model_cfg.is_standard_attention_layer(il)) continue;
        // Eligibility requires immutable source K, so the spill record stores
        // V only for every standard-attention layer.
        packed_targets.push_back(
            PackedTarget{&v_cache(il), false});
    }
    if (kvmem_mtp_tiered_) {
        // Eligibility also requires local-position MTP: raw MTP K is the
        // authority and only its position-free V enters the spill record.
        packed_targets.push_back(
            PackedTarget{&mtp_v_cache(), true});
    }
    if (packed_targets.empty()) return;

    const uint64_t slab_target_bytes = kvmem_io_slab_bytes();
    const uint64_t page_bytes = kvmem_kv_page_bytes();
    size_t candidate = 0;
    while (candidate < candidates.size()) {
        kvmem_reap_pending_writes(/*wait_all=*/false);
        while (kvmem_free_writeback_slabs_.empty()) {
            if (!kvmem_proactive_d2h_batches_.empty()) {
                kvmem_finish_proactive_d2h(/*wait_all=*/true);
                kvmem_reap_pending_writes(/*wait_all=*/false);
                continue;
            }
            if (kvmem_pending_writes_.empty()) {
                throw std::runtime_error(
                    "KVMem proactive write-back lost both pinned slabs");
            }
            kvmem_pending_writes_.front().future.wait();
            kvmem_reap_pending_writes(/*wait_all=*/false);
        }

        KvMemProactiveD2hBatch batch;
        batch.buffer =
            std::move(kvmem_free_writeback_slabs_.front());
        kvmem_free_writeback_slabs_.pop_front();
        batch.completed_pos = completed_pos;
        batch.submit_ns = kvmem_steady_ns();
        std::vector<uint64_t> block_offsets;
        while (candidate < candidates.size()) {
            const uint32_t id = candidates[candidate];
            const uint64_t bytes =
                kvmem_block_spill_bytes(blocks[id]);
            if (!batch.block_ids.empty() &&
                batch.bytes + bytes > slab_target_bytes) {
                break;
            }
            if (batch.bytes + bytes > batch.buffer->bytes) {
                throw std::runtime_error(
                    "KVMem proactive block exceeds its D2H slab");
            }
            if (kvmem_nvme_tier_) {
                const NvmeSlotPlacement placement =
                    kvmem_nvme_tier_->place_block(id);
                if (placement.slot < 0) {
                    throw std::runtime_error(
                        "KVMem proactive write-back exhausted the SSD arena");
                }
                batch.spans.push_back(
                    NvmeIoSpan{placement.slot, batch.bytes, bytes});
            } else {
                PinnedSlotPlacement placement =
                    kvmem_cpu_tier_->place_block(id);
                if (placement.slot < 0 ||
                    (kvmem_sparse_cpu_tier_ &&
                     !kvmem_reserve_cpu_slot(placement.slot))) {
                    if (placement.slot >= 0) {
                        kvmem_cpu_tier_->release_block(id);
                    }
                    throw std::runtime_error(
                        "KVMem proactive CPU write-back exhausted its "
                        "inclusive tier");
                }
                uint8_t *dst =
                    kvmem_cpu_slot_data(placement.slot);
                if (!dst) {
                    kvmem_cpu_tier_->release_block(id);
                    kvmem_release_cpu_slot(placement.slot);
                    throw std::runtime_error(
                        "KVMem proactive CPU write-back lost its slot");
                }
                batch.cpu_copies.push_back(
                    KvMemPendingWriteBatch::CpuCopy{
                        id, placement.slot, dst, batch.bytes, bytes});
                block_store_->set_block_io_in_flight(id, true);
            }
            block_offsets.push_back(batch.bytes);
            batch.block_ids.push_back(id);
            batch.bytes += bytes;
            ++candidate;
        }

        uint32_t pages_per_target = 0;
        for (uint32_t id : batch.block_ids) {
            const KvMemBlock &block = blocks[id];
            const uint32_t page_size = kv_pages_.page_size;
            const uint32_t first_page =
                block.orig_pos_start / page_size;
            const uint32_t last_page =
                (block.orig_pos_start + block.n_tokens - 1) /
                page_size;
            pages_per_target += last_page - first_page + 1;
        }
        batch.src_page_indices.resize(
            static_cast<size_t>(pages_per_target) *
            packed_targets.size());
        batch.dst_page_indices.resize(
            batch.src_page_indices.size());
        for (size_t t = 0; t < packed_targets.size(); ++t) {
            uint32_t page_entry = 0;
            for (size_t b = 0; b < batch.block_ids.size(); ++b) {
                const KvMemBlock &block =
                    blocks[batch.block_ids[b]];
                const uint32_t page_size = kv_pages_.page_size;
                const uint32_t first_page =
                    block.orig_pos_start / page_size;
                const uint32_t last_page =
                    (block.orig_pos_start + block.n_tokens - 1) /
                    page_size;
                const uint32_t block_pages =
                    last_page - first_page + 1;
                const uint64_t packed_dst_base =
                    block_offsets[b] / page_bytes +
                    static_cast<uint64_t>(t) * block_pages;
                for (uint32_t p = 0; p < block_pages; ++p) {
                    const size_t index =
                        t * pages_per_target + page_entry;
                    const uint32_t logical_page = first_page + p;
                    batch.src_page_indices[index] =
                        packed_targets[t].mtp
                            ? mtp_kv_pages_.pages[logical_page]
                            : kv_pages_.pages[logical_page];
                    batch.dst_page_indices[index] =
                        static_cast<int32_t>(packed_dst_base + p);
                    ++page_entry;
                }
            }
        }
        std::vector<DeviceTensor *> target_tensors;
        target_tensors.reserve(packed_targets.size());
        for (const PackedTarget &target : packed_targets) {
            target_tensors.push_back(target.tensor);
        }
        try {
            require_status(
                backend_.begin_kv_transfer_from_device());
            require_status(backend_.copy_packed_pages_to_host_async(
                batch.buffer->data, batch.bytes, target_tensors,
                batch.src_page_indices.data(),
                batch.dst_page_indices.data(),
                pages_per_target, page_bytes));
            require_status(
                backend_.record_kv_transfer_fence(batch.fence));
        } catch (...) {
            if (!kvmem_nvme_tier_) {
                for (const auto &copy : batch.cpu_copies) {
                    block_store_->set_block_io_in_flight(
                        copy.block_id, false);
                    if (kvmem_cpu_tier_->block_slot(copy.block_id) ==
                        copy.slot) {
                        kvmem_cpu_tier_->release_block(copy.block_id);
                        kvmem_release_cpu_slot(copy.slot);
                        block_store_->set_block_cpu_copy(
                            copy.block_id, -1);
                    }
                }
            }
            if (batch.buffer &&
                kvmem_free_writeback_slabs_.size() <
                    kvmem_writeback_slab_count_) {
                kvmem_free_writeback_slabs_.push_back(
                    std::move(batch.buffer));
            }
            throw;
        }

        kvmem_writeback_d2h_bytes_ += batch.bytes;
        kvmem_writeback_blocks_ +=
            static_cast<uint32_t>(batch.block_ids.size());
        ++kvmem_writeback_d2h_batches_;
        if (kvmem_perf_trace_flag()) {
            std::fprintf(
                stderr,
                "[kvmem-writeback] event=d2h_submit completed_pos=%u "
                "blocks=%zu bytes=%llu target=%s pending_d2h=%zu\n",
                completed_pos, batch.block_ids.size(),
                static_cast<unsigned long long>(batch.bytes),
                kvmem_nvme_tier_ ? "ssd" : "cpu",
                kvmem_proactive_d2h_batches_.size() + 1);
        }
        kvmem_proactive_d2h_batches_.push_back(
            std::move(batch));
    }
}

void QwenExecutor::kvmem_stage_out_cpu_batched(
        const std::vector<uint32_t> &block_ids) {
    if (!block_store_ || block_ids.empty()) return;
    if (!kvmem_cpu_tier_ || kvmem_nvme_tier_) {
        throw std::runtime_error(
            "KVMem batched CPU stage-out requires a CPU-only spill tier");
    }
    struct CpuStageOutTask {
        uint32_t block_id = 0;
        int32_t cpu_slot = -1;
        uint64_t offset = 0;
        uint64_t bytes = 0;
    };
    struct CpuStageOutBatch {
        std::unique_ptr<HostBuffer> buffer;
        std::vector<CpuStageOutTask> tasks;
        // Pageable metadata must remain alive until the copy-stream fence:
        // CUDA may defer the small index uploads that precede GPU gather.
        std::vector<int32_t> src_page_indices;
        std::vector<int32_t> dst_page_indices;
        std::unique_ptr<DeviceTransferFence> fence;
        uint64_t bytes = 0;
    };

    const bool perf_trace = kvmem_perf_trace_flag();
    const bool tm = kvmem_measure_timing_flag();
    const uint64_t total0 =
        (perf_trace || tm) ? kvmem_steady_ns() : 0;
    const uint64_t slab_target_bytes = kvmem_io_slab_bytes();
    const auto &blocks = block_store_->blocks();
    std::deque<CpuStageOutBatch> pending;
    CpuStageOutBatch building;
    uint32_t staged_out = 0;

    auto acquire_slab = [&](uint64_t required) {
        for (auto it = kvmem_free_stageout_slabs_.begin();
             it != kvmem_free_stageout_slabs_.end(); ++it) {
            if ((*it)->bytes >= required) {
                auto slab = std::move(*it);
                kvmem_free_stageout_slabs_.erase(it);
                return slab;
            }
        }
        return backend_.host_buffer(
            std::max<uint64_t>(required, slab_target_bytes),
            "kvmem_stageout_cpu_slab");
    };
    auto release_gpu_pages = [&](const KvMemBlock &block) {
        const uint32_t page_size = kv_pages_.page_size;
        const uint32_t first_page =
            block.orig_pos_start / page_size;
        const uint32_t last_page =
            (block.orig_pos_start + block.n_tokens - 1) /
            page_size;
        kv_pages_.release_logical_pages(
            backend_, first_page, last_page - first_page + 1);
        if (kvmem_mtp_tiered_) {
            mtp_kv_pages_.release_logical_pages(
                backend_, first_page, last_page - first_page + 1);
        }
        return last_page - first_page + 1;
    };
    auto finish_front = [&]() {
        if (pending.empty()) return;
        CpuStageOutBatch batch = std::move(pending.front());
        pending.pop_front();
        const uint64_t wait0 =
            perf_trace ? kvmem_steady_ns() : 0;
        require_status(
            backend_.wait_kv_transfer_fence(batch.fence));
        if (perf_trace) {
            kvmem_last_stage_out_perf_.async_backpressure_ns +=
                kvmem_steady_ns() - wait0;
        }
        const uint64_t scatter0 =
            perf_trace ? kvmem_steady_ns() : 0;
        const auto *slab =
            static_cast<const uint8_t *>(batch.buffer->data);
        std::vector<uint8_t *> scatter_dsts(batch.tasks.size());
        for (size_t i = 0; i < batch.tasks.size(); ++i) {
            scatter_dsts[i] =
                kvmem_cpu_slot_data(batch.tasks[i].cpu_slot);
            if (!scatter_dsts[i]) {
                throw std::runtime_error(
                    "KVMem batched CPU stage-out lost its destination slot");
            }
        }
        auto scatter_task = [&](size_t index) {
            const CpuStageOutTask &task = batch.tasks[index];
            std::memcpy(
                scatter_dsts[index],
                slab + task.offset,
                static_cast<size_t>(task.bytes));
        };
        const size_t scatter_workers = std::min(
            kvmem_cpu_stageout_threads(), batch.tasks.size());
        if (scatter_workers <= 1) {
            for (size_t i = 0; i < batch.tasks.size(); ++i) {
                scatter_task(i);
            }
        } else if (kvmem_cpu_worker_pool_) {
            kvmem_cpu_worker_pool_->run(
                batch.tasks.size(), scatter_workers, scatter_task);
        } else {
            std::atomic<size_t> next_task{0};
            auto worker = [&]() {
                for (;;) {
                    const size_t i =
                        next_task.fetch_add(
                            1, std::memory_order_relaxed);
                    if (i >= batch.tasks.size()) break;
                    scatter_task(i);
                }
            };
            std::vector<std::thread> workers;
            workers.reserve(scatter_workers - 1);
            for (size_t i = 1; i < scatter_workers; ++i) {
                workers.emplace_back(worker);
            }
            worker();
            for (std::thread &thread : workers) thread.join();
        }
        if (perf_trace) {
            const uint64_t elapsed =
                kvmem_steady_ns() - scatter0;
            kvmem_last_stage_out_perf_.cpu_copy_ns += elapsed;
            kvmem_last_stage_out_perf_.async_gather_ns += elapsed;
        }
        std::vector<std::pair<uint32_t, uint32_t>> release_ranges;
        release_ranges.reserve(batch.tasks.size());
        for (const CpuStageOutTask &task : batch.tasks) {
            const KvMemBlock &block = blocks[task.block_id];
            const uint32_t page_size = kv_pages_.page_size;
            const uint32_t first_page =
                block.orig_pos_start / page_size;
            const uint32_t last_page =
                (block.orig_pos_start + block.n_tokens - 1) /
                page_size;
            release_ranges.emplace_back(
                first_page, last_page - first_page + 1);
        }
        kv_pages_.release_logical_page_ranges(
            backend_, release_ranges);
        if (kvmem_mtp_tiered_) {
            mtp_kv_pages_.release_logical_page_ranges(
                backend_, release_ranges);
        }
        for (const CpuStageOutTask &task : batch.tasks) {
            const KvMemBlock &block = blocks[task.block_id];
            const uint32_t page_size = kv_pages_.page_size;
            const uint32_t first_page =
                block.orig_pos_start / page_size;
            const uint32_t last_page =
                (block.orig_pos_start + block.n_tokens - 1) /
                page_size;
            const uint32_t released_pages =
                last_page - first_page + 1;
            block_store_->set_block_tier(
                task.block_id, KvTier::CPU,
                task.cpu_slot, -1);
            ++staged_out;
            if (perf_trace) {
                ++kvmem_last_stage_out_perf_.blocks;
                ++kvmem_last_stage_out_perf_.cpu_blocks;
                kvmem_last_stage_out_perf_.bytes += task.bytes;
            }
            if (kvmem_tier_trace_enabled()) {
                std::fprintf(
                    stderr,
                    "[kvmem-tier] stage_out block=%u to=cpu "
                    "slot=%d bytes=%llu pages=%u mode=batched\n",
                    task.block_id, task.cpu_slot,
                    static_cast<unsigned long long>(task.bytes),
                    released_pages);
            }
        }
        if (batch.buffer &&
            kvmem_free_stageout_slabs_.size() < 2) {
            kvmem_free_stageout_slabs_.push_back(
                std::move(batch.buffer));
        }
    };
    auto submit_building = [&]() {
        if (building.tasks.empty()) return;
        const uint64_t submit0 =
            perf_trace ? kvmem_steady_ns() : 0;
        require_status(
            backend_.begin_kv_transfer_from_device());
        auto *slab =
            static_cast<uint8_t *>(building.buffer->data);
        struct PackedTarget {
            DeviceTensor *tensor = nullptr;
            bool mtp = false;
        };
        std::vector<PackedTarget> packed_targets;
        packed_targets.reserve(kvmem_raw_layers_.size() * 2 + 2);
        const QwenConfig &cfg = model_.config();
        for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
            if (!cfg.is_standard_attention_layer(il)) continue;
            if (!kvmem_immutable_source_k_) {
                packed_targets.push_back(
                    PackedTarget{&k_cache(il), false});
            }
            packed_targets.push_back(
                PackedTarget{&v_cache(il), false});
        }
        if (kvmem_mtp_tiered_) {
            if (!kvmem_mtp_local_positions_) {
                packed_targets.push_back(
                    PackedTarget{&mtp_k_cache(), true});
            }
            packed_targets.push_back(
                PackedTarget{&mtp_v_cache(), true});
        }
        if (packed_targets.empty()) {
            throw std::runtime_error(
                "KVMem packed CPU stage-out has no source tensors");
        }

        uint32_t pages_per_target = 0;
        for (const CpuStageOutTask &task : building.tasks) {
            const KvMemBlock &block = blocks[task.block_id];
            const uint32_t page_size = kv_pages_.page_size;
            const uint32_t first_page =
                block.orig_pos_start / page_size;
            const uint32_t last_page =
                (block.orig_pos_start + block.n_tokens - 1) /
                page_size;
            pages_per_target += last_page - first_page + 1;
        }
        building.src_page_indices.resize(
            static_cast<size_t>(pages_per_target) *
            packed_targets.size());
        building.dst_page_indices.resize(
            building.src_page_indices.size());
        const uint64_t page_bytes = kvmem_kv_page_bytes();
        for (size_t t = 0; t < packed_targets.size(); ++t) {
            uint32_t page_entry = 0;
            for (const CpuStageOutTask &task : building.tasks) {
                const KvMemBlock &block = blocks[task.block_id];
                const uint32_t page_size = kv_pages_.page_size;
                const uint32_t first_page =
                    block.orig_pos_start / page_size;
                const uint32_t last_page =
                    (block.orig_pos_start + block.n_tokens - 1) /
                    page_size;
                const uint32_t block_pages =
                    last_page - first_page + 1;
                if (task.offset % page_bytes != 0) {
                    throw std::runtime_error(
                        "KVMem packed CPU stage-out block is not "
                        "page-aligned");
                }
                const uint64_t packed_dst_base =
                    task.offset / page_bytes +
                    static_cast<uint64_t>(t) * block_pages;
                for (uint32_t p = 0; p < block_pages; ++p) {
                    const size_t index =
                        t * pages_per_target + page_entry;
                    const uint32_t logical_page = first_page + p;
                    building.src_page_indices[index] =
                        packed_targets[t].mtp
                            ? mtp_kv_pages_.pages[logical_page]
                            : kv_pages_.pages[logical_page];
                    building.dst_page_indices[index] =
                        static_cast<int32_t>(packed_dst_base + p);
                    ++page_entry;
                }
            }
        }
        std::vector<DeviceTensor *> target_tensors;
        target_tensors.reserve(packed_targets.size());
        for (const PackedTarget &target : packed_targets) {
            target_tensors.push_back(target.tensor);
        }
        require_status(backend_.copy_packed_pages_to_host_async(
            slab, building.bytes, target_tensors,
            building.src_page_indices.data(),
            building.dst_page_indices.data(),
            pages_per_target, page_bytes));
        require_status(
            backend_.record_kv_transfer_fence(building.fence));
        if (perf_trace) {
            kvmem_last_stage_out_perf_.async_submit_ns +=
                kvmem_steady_ns() - submit0;
            kvmem_last_stage_out_perf_.async_submitted_bytes +=
                building.bytes;
            ++kvmem_last_stage_out_perf_.async_batches;
        }
        pending.push_back(std::move(building));
        building = CpuStageOutBatch{};
        // Once B is queued behind A, waiting for A exposes only the part not
        // already hidden by B's canonicalization/enqueue. CPU scatter of A
        // then overlaps the copy-stream transfer of B.
        if (pending.size() >= 2) finish_front();
    };
    auto place_cpu = [&](uint32_t block_id,
                         uint64_t bytes) -> int32_t {
        PinnedSlotPlacement placement;
        if (kvmem_sparse_cpu_tier_) {
            if (kvmem_cpu_budget_has(bytes)) {
                placement =
                    kvmem_cpu_tier_->place_block(block_id);
                if (placement.slot >= 0 &&
                    !kvmem_reserve_cpu_slot(placement.slot)) {
                    kvmem_cpu_tier_->release_block(block_id);
                    placement = PinnedSlotPlacement{};
                }
            }
        } else {
            placement =
                kvmem_cpu_tier_->place_block(block_id);
        }
        if (placement.slot < 0) {
            throw std::runtime_error(
                "kvmem: CPU-only opt_2 batch cannot reserve a spill "
                "slot; raise --kvmem-cpu-gb or lower --ctx");
        }
        return placement.slot;
    };

    try {
        for (uint32_t block_id : block_ids) {
            if (block_id >= blocks.size()) continue;
            const KvMemBlock &block = blocks[block_id];
            if (!kvmem_block_pages_resident(block)) continue;
            const uint64_t bytes =
                kvmem_block_spill_bytes(block);
            const int32_t cached =
                kvmem_cpu_tier_->block_slot(block_id);
            if (cached >= 0 && kvmem_cpu_slot_data(cached)) {
                const uint32_t released_pages =
                    release_gpu_pages(block);
                block_store_->set_block_tier(
                    block_id, KvTier::CPU, cached, -1);
                ++staged_out;
                if (perf_trace) {
                    ++kvmem_last_stage_out_perf_.blocks;
                    ++kvmem_last_stage_out_perf_.cpu_blocks;
                    ++kvmem_last_stage_out_perf_.clean_blocks;
                    kvmem_last_stage_out_perf_.clean_bytes_avoided +=
                        bytes;
                }
                if (kvmem_tier_trace_enabled()) {
                    std::fprintf(
                        stderr,
                        "[kvmem-tier] stage_out_clean block=%u "
                        "to=cpu cpu_slot=%d avoided_bytes=%llu "
                        "pages=%u\n",
                        block_id, cached,
                        static_cast<unsigned long long>(bytes),
                        released_pages);
                }
                continue;
            }
            if (!building.tasks.empty() &&
                building.bytes + bytes > slab_target_bytes) {
                submit_building();
            }
            if (!building.buffer) {
                building.buffer = acquire_slab(
                    std::max<uint64_t>(bytes, slab_target_bytes));
            }
            const int32_t cpu_slot =
                place_cpu(block_id, bytes);
            const uint64_t canonical0 =
                perf_trace ? kvmem_steady_ns() : 0;
            kvmem_canonicalize_block_for_tier(block_id);
            if (kvmem_mtp_local_positions_ &&
                !kvmem_block_mtp_pages_resident(block)) {
                throw std::runtime_error(
                    "KVMem local-position MTP attempted to stage out "
                    "a block before its V pages were primed");
            }
            if (perf_trace) {
                kvmem_last_stage_out_perf_.
                    canonicalize_and_d2h_ns +=
                    kvmem_steady_ns() - canonical0;
            }
            building.tasks.push_back(
                CpuStageOutTask{
                    block_id, cpu_slot, building.bytes, bytes});
            building.bytes += bytes;
        }
        submit_building();
        while (!pending.empty()) finish_front();
    } catch (...) {
        // A fence may still own one of the pinned slabs. Drain the copy stream
        // before local batch destructors release host memory.
        (void)backend_.wait_kv_transfer();
        throw;
    }

    if (perf_trace) {
        kvmem_last_stage_out_perf_.total_ns =
            kvmem_steady_ns() - total0;
    }
    if (tm) {
        KvMemTimingTotals &t = kvmem_timing_totals();
        t.stage_out_ns.fetch_add(
            kvmem_steady_ns() - total0,
            std::memory_order_relaxed);
        t.stage_out_calls.fetch_add(
            1, std::memory_order_relaxed);
        t.stage_out_blocks.fetch_add(
            staged_out, std::memory_order_relaxed);
    }
}

void QwenExecutor::kvmem_stage_out(const std::vector<uint32_t> &block_ids) {
    kvmem_last_stage_out_perf_ = KvMemStageOutPerf{};
    if (!block_store_ || block_ids.empty()) return;
    if (!kvmem_cpu_tier_ && !kvmem_nvme_tier_) return;
    if (block_store_->config().optimization_level >=
            KvMemOptimizationLevel::Opt2 &&
        kvmem_cpu_tier_ && !kvmem_nvme_tier_) {
        if (kvmem_prefill_writeback_enabled_) {
            // A CPU slot is not authoritative until both the proactive D2H
            // fence and the pinned-to-pageable scatter have completed. Drain
            // that bounded pipeline before the clean-eviction fast path
            // inspects the CPU tier. Normally both phases were hidden by the
            // following prefill chunk and these waits are already ready.
            kvmem_finish_proactive_d2h(/*wait_all=*/true);
            kvmem_reap_pending_writes(/*wait_all=*/true);
        }
        kvmem_stage_out_cpu_batched(block_ids);
        return;
    }
    const bool perf_trace = kvmem_perf_trace_flag();
    const bool tm = kvmem_measure_timing_flag();
    const uint64_t t_out0 = tm ? kvmem_steady_ns() : 0;
    uint32_t staged_out = 0;
    const auto &blocks = block_store_->blocks();
    const bool inclusive =
        block_store_->config().optimization_level >=
            KvMemOptimizationLevel::Opt2 &&
        kvmem_nvme_tier_ != nullptr;
    if (inclusive) {
        // Convert any prefill-overlapped D2H slabs into SSD-owned in-flight
        // writes before eviction inspects ssd_clean/in_flight. Once this
        // returns, a pressure victim can release its GPU page without another
        // D2H because the pinned slab is the durable temporary owner.
        kvmem_finish_proactive_d2h(/*wait_all=*/true);
        kvmem_reap_pending_writes(/*wait_all=*/false);
    }
    const uint64_t write_slab_target_bytes = kvmem_io_slab_bytes();
    std::shared_ptr<std::vector<uint8_t>> write_buffer;
    std::vector<NvmeIoSpan> write_spans;
    std::vector<uint32_t> write_blocks;
    auto submit_write_slab = [&]() {
        if (!write_buffer || write_spans.empty()) return;
        const uint64_t t0 = kvmem_steady_ns();
        kvmem_submit_write_batch(
            std::move(write_buffer), std::move(write_spans),
            std::move(write_blocks));
        kvmem_last_stage_out_perf_.async_submit_ns +=
            kvmem_steady_ns() - t0;
        write_buffer.reset();
        write_spans.clear();
        write_blocks.clear();
    };
    auto queue_ssd_write = [&](uint32_t block_id, const uint8_t *src,
                               uint64_t bytes) {
        if (!kvmem_nvme_tier_) {
            throw std::runtime_error(
                "KVMem opt_2 async stage-out has no NVMe tier");
        }
        const NvmeSlotPlacement p =
            kvmem_nvme_tier_->place_block(block_id);
        if (p.slot < 0) {
            throw std::runtime_error(
                "KVMem opt_2 inclusive NVMe tier is full");
        }
        if (write_buffer && !write_spans.empty() &&
            write_buffer->size() + bytes > write_slab_target_bytes) {
            submit_write_slab();
        }
        if (!write_buffer) {
            if (kvmem_free_write_slabs_.empty() &&
                !kvmem_pending_writes_.empty()) {
                const uint64_t wait0 =
                    perf_trace ? kvmem_steady_ns() : 0;
                if (kvmem_pending_writes_.size() >=
                    kvmem_write_queue_depth()) {
                    kvmem_pending_writes_.front().future.wait();
                }
                kvmem_reap_pending_writes(/*wait_all=*/false);
                if (perf_trace) {
                    kvmem_last_stage_out_perf_.async_backpressure_ns +=
                        kvmem_steady_ns() - wait0;
                }
            }
            if (!kvmem_free_write_slabs_.empty()) {
                write_buffer =
                    std::move(kvmem_free_write_slabs_.front());
                kvmem_free_write_slabs_.pop_front();
                write_buffer->clear();
            } else {
                write_buffer =
                    std::make_shared<std::vector<uint8_t>>();
            }
            write_buffer->reserve(static_cast<size_t>(
                std::max<uint64_t>(write_slab_target_bytes, bytes)));
        }
        const uint64_t offset = write_buffer->size();
        const uint64_t t_gather0 =
            perf_trace ? kvmem_steady_ns() : 0;
        write_buffer->insert(
            write_buffer->end(), src, src + static_cast<size_t>(bytes));
        if (perf_trace) {
            kvmem_last_stage_out_perf_.async_gather_ns +=
                kvmem_steady_ns() - t_gather0;
        }
        write_spans.push_back(NvmeIoSpan{p.slot, offset, bytes});
        write_blocks.push_back(block_id);
        block_store_->set_block_ssd_backing(block_id, p.slot, false);
        block_store_->set_block_io_in_flight(block_id, true);
        return p.slot;
    };
    auto release_gpu_pages = [&](const KvMemBlock &block) {
        const uint32_t page_size = kv_pages_.page_size;
        const uint32_t first_page = block.orig_pos_start / page_size;
        const uint32_t last_page =
            (block.orig_pos_start + block.n_tokens - 1) / page_size;
        kv_pages_.release_logical_pages(
            backend_, first_page, last_page - first_page + 1);
        if (kvmem_mtp_tiered_) {
            mtp_kv_pages_.release_logical_pages(
                backend_, first_page, last_page - first_page + 1);
        }
        return last_page - first_page + 1;
    };
    for (uint32_t block_id : block_ids) {
        if (block_id >= blocks.size()) continue;
        const KvMemBlock &block = blocks[block_id];
        if (!kvmem_block_pages_resident(block)) continue;
        const uint64_t src_bytes = kvmem_block_spill_bytes(block);
        if (inclusive && (block.ssd_clean || block.in_flight)) {
            if (!kvmem_nvme_tier_ || block.nvme_slot < 0 ||
                kvmem_nvme_tier_->block_slot(block_id) != block.nvme_slot) {
                throw std::runtime_error(
                    "KVMem opt_2 block has clean/pending SSD backing without "
                    "a matching inclusive slot");
            }
            int32_t cpu_slot = -1;
            if (kvmem_cpu_tier_) {
                const int32_t cached =
                    kvmem_cpu_tier_->block_slot(block_id);
                if (cached >= 0 && block.cpu_slot == cached &&
                    kvmem_cpu_slot_data(cached)) {
                    cpu_slot = cached;
                }
            }
            const uint32_t released_pages = release_gpu_pages(block);
            block_store_->set_block_tier(
                block_id, cpu_slot >= 0 ? KvTier::CPU : KvTier::SSD,
                cpu_slot, block.nvme_slot);
            ++staged_out;
            if (perf_trace) {
                ++kvmem_last_stage_out_perf_.blocks;
                ++kvmem_last_stage_out_perf_.clean_blocks;
                kvmem_last_stage_out_perf_.clean_bytes_avoided += src_bytes;
                if (cpu_slot >= 0) {
                    ++kvmem_last_stage_out_perf_.cpu_blocks;
                } else {
                    ++kvmem_last_stage_out_perf_.nvme_blocks;
                }
            }
            if (kvmem_tier_trace_enabled()) {
                std::fprintf(
                    stderr,
                    "[kvmem-tier] stage_out_clean block=%u to=%s "
                    "cpu_slot=%d nvme_slot=%d avoided_bytes=%llu pages=%u\n",
                    block_id, cpu_slot >= 0 ? "cpu" : "nvme", cpu_slot,
                    block.nvme_slot,
                    static_cast<unsigned long long>(src_bytes),
                    released_pages);
            }
            continue;
        }
        // D2H into pinned host memory so the block's K/V page copies queue
        // asynchronously on the copy stream (a pageable buffer would serialize
        // them through the driver's bounce buffer).
        uint8_t *src = kvmem_ensure_stage_pinned(src_bytes);
        // Standard K is omitted from immutable spill records. Legacy tiered MTP
        // K still needs independent canonicalization; local-position MTP skips
        // it and spills V only because raw MTP K is authoritative elsewhere.
        const uint64_t t_d2h0 =
            perf_trace ? kvmem_steady_ns() : 0;
        kvmem_canonicalize_block_for_tier(block_id);
        if (kvmem_mtp_local_positions_ &&
            !kvmem_block_mtp_pages_resident(block)) {
            throw std::runtime_error(
                "KVMem local-position MTP attempted to stage out a block "
                "before its V pages were primed");
        }
        require_status(backend_.begin_kv_transfer_from_device());
        kvmem_copy_block_to_host_ptr(block, src);
        require_status(backend_.wait_kv_transfer());
        if (perf_trace) {
            kvmem_last_stage_out_perf_.canonicalize_and_d2h_ns +=
                kvmem_steady_ns() - t_d2h0;
        }
        bool placed = false;
        int32_t cpu_slot = -1;
        int32_t nvme_slot = -1;
        if (inclusive) {
            nvme_slot = queue_ssd_write(block_id, src, src_bytes);
        }
        if (kvmem_cpu_tier_) {
            PinnedSlotPlacement placement;
            if (kvmem_sparse_cpu_tier_) {
                if (kvmem_cpu_budget_has(src_bytes)) {
                    placement = kvmem_cpu_tier_->place_block(block_id);
                    if (placement.slot >= 0 &&
                        !kvmem_reserve_cpu_slot(placement.slot)) {
                        kvmem_cpu_tier_->release_block(block_id);
                        placement = PinnedSlotPlacement{};
                    }
                } else if (
                    kvmem_nvme_tier_ &&
                    block_store_->config().optimization_level >=
                        KvMemOptimizationLevel::Opt1) {
                    // Shared raw-K + spill accounting can exhaust the physical
                    // CPU byte budget before the tier's logical slot range is
                    // full. Reuse a cold resident's already allocated buffer;
                    // a colder candidate is rejected and falls through to SSD.
                    placement =
                        kvmem_cpu_tier_->place_block_replacing(block_id);
                }
            } else if (kvmem_nvme_tier_) {
                placement = kvmem_cpu_tier_->place_block_evicting(block_id);
            } else {
                placement = kvmem_cpu_tier_->place_block(block_id);
            }
            if (placement.slot >= 0) {
                if (placement.evicted_block >= 0) {
                    const uint32_t victim =
                        static_cast<uint32_t>(placement.evicted_block);
                    const uint64_t victim_bytes =
                        kvmem_block_spill_bytes(blocks[victim]);
                    const uint8_t *victim_src =
                        kvmem_cpu_slot_data(placement.slot);
                    if (!victim_src) {
                        throw std::runtime_error(
                            "KVMem CPU eviction lost its slot backing");
                    }
                    const KvMemBlock &victim_block = blocks[victim];
                    const bool victim_backed =
                        inclusive &&
                        (victim_block.ssd_clean ||
                         victim_block.in_flight) &&
                        victim_block.nvme_slot >= 0 &&
                        kvmem_nvme_tier_->block_slot(victim) ==
                            victim_block.nvme_slot;
                    if (!victim_backed) {
                        const uint64_t t_nvme0 =
                            perf_trace ? kvmem_steady_ns() : 0;
                        kvmem_nvme_tier_->write_block(
                            victim, victim_src, victim_bytes);
                        if (perf_trace) {
                            kvmem_last_stage_out_perf_.nvme_write_ns +=
                                kvmem_steady_ns() - t_nvme0;
                        }
                        if (inclusive) {
                            block_store_->set_block_ssd_backing(
                                victim,
                                kvmem_nvme_tier_->block_slot(victim), true);
                        }
                    }
                    if (kvmem_tier_trace_enabled()) {
                        std::fprintf(stderr,
                                     "[kvmem-tier] cpu_evict block=%u to=nvme "
                                     "slot=%d bytes=%llu ssd_write=%d\n",
                                     victim, kvmem_nvme_tier_->block_slot(victim),
                                     static_cast<unsigned long long>(victim_bytes),
                                     victim_backed ? 0 : 1);
                    }
                    if (inclusive) {
                        block_store_->set_block_cpu_copy(victim, -1);
                        if (kvmem_block_pages_resident(victim_block)) {
                            block_store_->set_block_tier(
                                victim, KvTier::GPU);
                        } else {
                            block_store_->set_block_tier(
                                victim, KvTier::SSD, -1,
                                kvmem_nvme_tier_->block_slot(victim));
                        }
                    } else {
                        block_store_->set_block_tier(
                            victim, KvTier::SSD, -1,
                            kvmem_nvme_tier_->block_slot(victim));
                    }
                }
                uint8_t *dst = kvmem_cpu_slot_data(placement.slot);
                if (!dst) {
                    throw std::runtime_error(
                        "KVMem CPU tier slot has no writable backing");
                }
                const uint64_t t_copy0 =
                    perf_trace ? kvmem_steady_ns() : 0;
                std::memcpy(dst, src, src_bytes);
                if (perf_trace) {
                    kvmem_last_stage_out_perf_.cpu_copy_ns +=
                        kvmem_steady_ns() - t_copy0;
                }
                placed = true;
                cpu_slot = placement.slot;
            }
        }
        if (!placed && kvmem_nvme_tier_) {
            if (!inclusive) {
                const uint64_t t_nvme0 =
                    perf_trace ? kvmem_steady_ns() : 0;
                kvmem_nvme_tier_->write_block(block_id, src, src_bytes);
                if (perf_trace) {
                    kvmem_last_stage_out_perf_.nvme_write_ns +=
                        kvmem_steady_ns() - t_nvme0;
                }
            }
            placed = true;
            nvme_slot = kvmem_nvme_tier_->block_slot(block_id);
        }
        if (!placed) {
            // The spill tier is full and this block cannot be evicted from the
            // bounded GPU pool. Silently skipping (the old behavior) left the
            // block's GPU pages allocated -> the pool filled monotonically over
            // a long prefill and later threw the cryptic "local KVMem GPU page
            // pool exhausted". The configure-time capacity check now guarantees
            // CPU+NVMe can hold (ctx - budget) blocks, so this is unreachable in
            // a well-formed config; if it ever fires it is a real accounting
            // bug, so fail loud with the actionable knobs instead of leaking.
            if (kvmem_gpu_page_pool_) {
                throw std::runtime_error(
                    "kvmem: KV spill tier full during stage-out (block=" +
                    std::to_string(block_id) + "); the bounded GPU page pool "
                    "cannot free it because CPU+NVMe are exhausted. Raise "
                    "--kvmem-cpu-gb and/or add --kvmem-nvme-dir + "
                    "--kvmem-nvme-gb, or lower --ctx / raise --kvmem-budget.");
            }
            continue;
        }
        ++staged_out;
        if (perf_trace) {
            ++kvmem_last_stage_out_perf_.blocks;
            kvmem_last_stage_out_perf_.bytes += src_bytes;
            if (cpu_slot >= 0) {
                ++kvmem_last_stage_out_perf_.cpu_blocks;
            } else {
                ++kvmem_last_stage_out_perf_.nvme_blocks;
            }
        }
        const uint32_t released_pages = release_gpu_pages(block);
        block_store_->set_block_tier(
            block_id, cpu_slot >= 0 ? KvTier::CPU : KvTier::SSD,
            cpu_slot, nvme_slot);
        if (kvmem_tier_trace_enabled()) {
            std::fprintf(stderr,
                         "[kvmem-tier] stage_out block=%u to=%s slot=%d bytes=%llu pages=%u\n",
                         block_id, cpu_slot >= 0 ? "cpu" : "nvme",
                         cpu_slot >= 0 ? cpu_slot : nvme_slot,
                         static_cast<unsigned long long>(src_bytes),
                         released_pages);
        }
    }
    submit_write_slab();
    if (perf_trace) {
        kvmem_last_stage_out_perf_.total_ns =
            kvmem_steady_ns() - t_out0;
    }
    if (tm) {
        KvMemTimingTotals &t = kvmem_timing_totals();
        t.stage_out_ns.fetch_add(kvmem_steady_ns() - t_out0,
                                 std::memory_order_relaxed);
        t.stage_out_calls.fetch_add(1, std::memory_order_relaxed);
        t.stage_out_blocks.fetch_add(staged_out, std::memory_order_relaxed);
    }
}

void QwenExecutor::kvmem_assemble(const KvMemPlan &plan) {
    const bool tm = kvmem_measure_timing_flag();
    const uint64_t t_asm0 = tm ? kvmem_steady_ns() : 0;
    kvmem_assembly_raw_gather_ns_ = 0;
    kvmem_assembly_raw_h2d_submit_ns_ = 0;
    kvmem_assembly_raw_h2d_wait_ns_ = 0;
    kvmem_assembly_raw_bytes_ = 0;
    kvmem_assembly_raw_batches_ = 0;
    const QwenConfig &cfg = model_.config();
    kvmem_ensure_rope_sincos_table();
    const uint32_t page_size = kv_pages_.page_size;
    const uint32_t per_pos =
        static_cast<uint32_t>(cfg.n_kv_heads) * cfg.head_dim;
    if (kvmem_mtp_local_positions_ && cfg.n_ctx_train > 0 &&
        static_cast<uint64_t>(plan.total_window_tokens) +
                block_store_->config().gen_budget >
            cfg.n_ctx_train) {
        throw std::runtime_error(
            "KVMem assembled window plus generation reserve exceeds the "
            "model context limit in local-position MTP mode");
    }

    // Build the window page-index list = each selected block's ORIGINAL
    // physical pages, in window (ascending block-id) order, packed contiguously
    // from window position 0. No-copy: these alias the same physical pages the
    // block already occupies in kv_pages_; the window is just a pointer
    // reordering. baked_pos in the plan tells us where the block's K currently
    // sits so re-RoPE can de-rotate from there.
    window_pages_host_.clear();
    // MTP draft mirror: build the same window over the MTP draft head's own KV
    // cache (mtp_kv_pages_). Only when the MTP scratch is live (spec decode); a
    // plain kvmem decode never touches the MTP cache.
    //
    // The mirror requires every selected block's MTP pages to be primed. The MTP
    // prefix is primed lazily, AFTER each prefill chunk's main forward returns
    // (prime_mtp_prefix uses that chunk's batch hidden states), so during prefill
    // mtp_prefix_len_ lags the registered position: a mid-prefill reselect fired
    // by kvmem_maybe_prefill_offload registers blocks (from the freshly-written
    // MAIN cache) whose MTP pages do not exist yet. That mid-prefill mirror is
    // throwaway anyway -- it is only consumed by the decode-phase windowed draft,
    // and it is rebuilt at the post-prefill reselect once priming has caught up.
    // So gate the mirror on the MTP cache covering all registered blocks; when it
    // lags, skip the build (the stale mirror is never read before the rebuild).
    const bool build_mtp_window = kvmem_mtp_prefix_covers_registered();
    if (build_mtp_window) mtp_window_pages_host_.clear();
    const auto &blocks = block_store_->blocks();
    // Per-window-block metadata for the cumulative-attention selection signal:
    // window slot w -> (block_id, first window pos, token count). Built in the
    // same window order as the page list so kbar slot w lines up with accum[w].
    bs_window_block_ids_.clear();
    bs_win_base_host_.clear();
    bs_blk_tokens_host_.clear();
    for (const KvMemRemap &rm : plan.remaps) {
        const KvMemBlock &b = blocks[rm.block_id];
        // The block's original logical positions are
        // [orig_pos_start .. orig_pos_start + n_tokens). Map each through the
        // live kv_pages_ table to its physical page, and lay those physical
        // pages into the window in order. Because blocks are block_tokens-
        // aligned and the window packs them contiguously, a block that is not
        // page-aligned in the window would split a physical page across two
        // logical window slots — guard against it (v1 requires block_tokens to
        // be a multiple of page_size so window slots stay page-aligned).
        const uint32_t first_logical = b.orig_pos_start;
        const uint32_t last_logical = b.orig_pos_start + b.n_tokens - 1;
        const uint32_t first_page = first_logical / page_size;
        const uint32_t last_page = last_logical / page_size;
        for (uint32_t lp = first_page; lp <= last_page; ++lp) {
            if (lp >= kv_pages_.pages.size()) {
                throw std::runtime_error(
                    "block-sparse window references unallocated KV page");
            }
            if (kv_pages_.pages[lp] < 0) {
                throw std::runtime_error(
                    "block-sparse window references offloaded KV page");
            }
            window_pages_host_.push_back(kv_pages_.pages[lp]);
            if (build_mtp_window) {
                if (lp >= mtp_kv_pages_.pages.size() ||
                    mtp_kv_pages_.pages[lp] < 0) {
                    throw std::runtime_error(
                        "block-sparse MTP window references unprimed KV page");
                }
                mtp_window_pages_host_.push_back(mtp_kv_pages_.pages[lp]);
            }
        }
        bs_window_block_ids_.push_back(rm.block_id);
        bs_win_base_host_.push_back(rm.to_base);
        bs_blk_tokens_host_.push_back(static_cast<int32_t>(b.n_tokens));
    }
    window_page_count_ = static_cast<uint32_t>(window_pages_host_.size());
    window_query_pos_ = plan.total_window_tokens;
    sync_window_pages_device(window_page_count_);
    if (build_mtp_window) {
        mtp_window_page_count_ =
            static_cast<uint32_t>(mtp_window_pages_host_.size());
        sync_mtp_window_pages_device(mtp_window_page_count_);
    }
    uint64_t t_pages = 0;
    if (tm) {
        require_status(backend_.synchronize());
        t_pages = kvmem_steady_ns();
    }

    // Re-RoPE each moved block in place, per standard attention layer. Skipped
    // blocks (already baked at their window slot) issue no kernel. Collect all
    // moved (non-skip) blocks once and upload their {to_base, from_base,
    // n_tokens} to the device, then issue ONE batched launch per standard
    // attention layer (its grid covers every moved block at once) instead of a
    // per-(layer,block) launch storm. The window page table addresses the
    // physical pages, so the kernel maps window slot to_base+tok ->
    // page_indices[(to_base+tok)/page_size] just like append; moved blocks
    // occupy disjoint physical pages, so the batch races nowhere and is
    // byte-identical to the per-block loop.
    bs_remap_to_host_.clear();
    bs_remap_from_host_.clear();
    bs_remap_ntok_host_.clear();
    uint32_t max_block_tokens = 0;
    std::vector<const KvMemRemap *> raw_refreshes;
    uint32_t main_reused_blocks = 0;
    const uint32_t standard_remap_layers =
        count_standard_attention_layers(cfg, weights_.n_layers());
    for (const KvMemRemap &rm : plan.remaps) {
        if (rm.skip && rm.raw_refresh) {
            throw std::runtime_error(
                "KVMem invalid remap: raw-K refresh was hidden by skip");
        }
        if (kvmem_immutable_source_k_ &&
            !rm.working_k_resident &&
            (rm.skip || !rm.raw_refresh)) {
            throw std::runtime_error(
                "KVMem cold immutable block is missing raw-K reconstruction");
        }
        if (rm.skip) {
            ++main_reused_blocks;
            continue;
        }
        if (kvmem_rope_sincos_ &&
            (rm.to_base < 0 ||
             static_cast<uint64_t>(rm.to_base) + rm.n_tokens >
                 kvmem_rope_table_positions_)) {
            throw std::runtime_error(
                "KVMem destination position exceeds the assembly RoPE table");
        }
        const bool table_source_oob =
            kvmem_rope_sincos_ &&
            (rm.from_base < 0 ||
             static_cast<uint64_t>(rm.from_base) + rm.n_tokens >
                 kvmem_rope_table_positions_);
        if (rm.raw_refresh || table_source_oob) {
            raw_refreshes.push_back(&rm);
            continue;
        }
        trace_rope_position_if_out_of_range(
            "kvmem_assemble.main.from", rm.from_base, rm.n_tokens,
            cfg.n_ctx_train, -1, standard_remap_layers);
        trace_rope_position_if_out_of_range(
            "kvmem_assemble.main.to", rm.to_base, rm.n_tokens,
            cfg.n_ctx_train, -1, standard_remap_layers);
        bs_remap_to_host_.push_back(rm.to_base);
        bs_remap_from_host_.push_back(rm.from_base);
        bs_remap_ntok_host_.push_back(static_cast<int32_t>(rm.n_tokens));
        max_block_tokens = std::max(max_block_tokens, rm.n_tokens);
    }
    const uint32_t n_moved = static_cast<uint32_t>(bs_remap_to_host_.size());
    if (main_reused_blocks + n_moved + raw_refreshes.size() !=
        plan.remaps.size()) {
        throw std::runtime_error(
            "KVMem main-K assembly plan does not cover every selected block");
    }
    // QW3_KVMEM_NO_REROPE (experiment): skip the position-collapse re-RoPE so each
    // selected block keeps its ORIGINAL-position rotation. The windowed attention
    // then runs with the true query position (driven from base_pos/position_ in the
    // forward paths, not window_query_pos_), so relative distances are preserved.
    // Slots/page-table stay contiguous (addressing only). Default OFF -> byte-identical.
    if (!raw_refreshes.empty()) {
        if (kvmem_no_rerope_) {
            throw std::runtime_error(
                "immutable raw-K refresh is incompatible with "
                "QW3_KVMEM_NO_REROPE");
        }
        kvmem_materialize_raw_k(raw_refreshes);
    }
    if (n_moved > 0 && !kvmem_no_rerope_) {
        if (n_moved > bs_remap_capacity_) {
            bs_remap_capacity_ = n_moved;
            bs_remap_to_dev_ = backend_.tensor_i32(bs_remap_capacity_, "bs_remap_to");
            bs_remap_from_dev_ =
                backend_.tensor_i32(bs_remap_capacity_, "bs_remap_from");
            bs_remap_ntok_dev_ =
                backend_.tensor_i32(bs_remap_capacity_, "bs_remap_ntok");
        }
        require_status(backend_.copy_i32_from_host(
            *bs_remap_to_dev_, 0, bs_remap_to_host_.data(), n_moved));
        require_status(backend_.copy_i32_from_host(
            *bs_remap_from_dev_, 0, bs_remap_from_host_.data(), n_moved));
        require_status(backend_.copy_i32_from_host(
            *bs_remap_ntok_dev_, 0, bs_remap_ntok_host_.data(), n_moved));
        for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
            if (!cfg.is_standard_attention_layer(il)) continue;
            require_status(backend_.rope_block_remap_paged_batched_device(
                k_cache(il), n_moved, max_block_tokens,
                cfg.n_kv_heads, per_pos,
                cfg.head_dim, cfg.rope_dim, *bs_remap_to_dev_,
                *bs_remap_from_dev_, *bs_remap_ntok_dev_, *window_pages_device_,
                page_size, cfg.rope_theta, kvmem_rope_sincos_.get(),
                kvmem_rope_table_positions_));
        }
    }
    // MTP draft head: in immutable local-position mode, never transform a
    // previously rotated working K. Rebuild every selected K directly from the
    // position-free CPU authority at rm.to_base. This also makes the result
    // independent of how often the block was selected or spilled. The legacy
    // path below retains the MTP-specific baked-position bookkeeping.
    uint32_t mtp_rebuild_blocks = 0;
    uint32_t mtp_inplace_blocks = 0;
    if (build_mtp_window && kvmem_mtp_local_positions_) {
        kvmem_sync_mtp_baked_pos();
        KvMemPlan mtp_rebuild_plan;
        mtp_rebuild_plan.remaps.reserve(plan.remaps.size());
        // Raw MTP materialization uses the persistent bs_mtp_remap_* vectors
        // as its own per-batch scratch. Keep the incremental lists local until
        // that path completes, then publish them for the in-place launch.
        std::vector<int32_t> mtp_incremental_to;
        std::vector<int32_t> mtp_incremental_from;
        std::vector<int32_t> mtp_incremental_ntok;
        mtp_incremental_to.reserve(plan.remaps.size());
        mtp_incremental_from.reserve(plan.remaps.size());
        mtp_incremental_ntok.reserve(plan.remaps.size());
        uint32_t max_mtp_block_tokens = 0;
        for (const KvMemRemap &rm : plan.remaps) {
            const int64_t mtp_from = mtp_baked_pos_[rm.block_id];
            const bool cold_rebuild =
                !rm.working_k_resident || rm.raw_refresh;
            if (!cold_rebuild &&
                mtp_from == static_cast<int64_t>(rm.to_base)) {
                continue;
            }
            const bool source_out_of_range =
                cfg.n_ctx_train > 0 &&
                (mtp_from < 0 ||
                 static_cast<uint64_t>(mtp_from) + rm.n_tokens >
                     cfg.n_ctx_train);
            const bool exact_rebuild =
                !kvmem_mtp_incremental_assembly_ ||
                kvmem_no_rerope_ ||
                cold_rebuild ||
                source_out_of_range;
            if (exact_rebuild) {
                mtp_rebuild_plan.remaps.push_back(rm);
                continue;
            }
            trace_rope_position_if_out_of_range(
                "kvmem_assemble.mtp_incremental.from", mtp_from,
                rm.n_tokens, cfg.n_ctx_train);
            trace_rope_position_if_out_of_range(
                "kvmem_assemble.mtp_incremental.to", rm.to_base,
                rm.n_tokens, cfg.n_ctx_train);
            mtp_incremental_to.push_back(rm.to_base);
            mtp_incremental_from.push_back(
                static_cast<int32_t>(mtp_from));
            mtp_incremental_ntok.push_back(
                static_cast<int32_t>(rm.n_tokens));
            max_mtp_block_tokens =
                std::max(max_mtp_block_tokens, rm.n_tokens);
        }
        mtp_rebuild_blocks =
            static_cast<uint32_t>(mtp_rebuild_plan.remaps.size());
        kvmem_materialize_raw_mtp_k(mtp_rebuild_plan);
        bs_mtp_remap_to_host_ = std::move(mtp_incremental_to);
        bs_mtp_remap_from_host_ = std::move(mtp_incremental_from);
        bs_mtp_remap_ntok_host_ = std::move(mtp_incremental_ntok);
        mtp_inplace_blocks =
            static_cast<uint32_t>(bs_mtp_remap_to_host_.size());
        if (mtp_inplace_blocks > 0) {
            if (mtp_inplace_blocks > bs_mtp_remap_capacity_) {
                bs_mtp_remap_capacity_ = mtp_inplace_blocks;
                bs_mtp_remap_to_dev_ = backend_.tensor_i32(
                    bs_mtp_remap_capacity_, "bs_mtp_remap_to");
                bs_mtp_remap_from_dev_ = backend_.tensor_i32(
                    bs_mtp_remap_capacity_, "bs_mtp_remap_from");
                bs_mtp_remap_ntok_dev_ = backend_.tensor_i32(
                    bs_mtp_remap_capacity_, "bs_mtp_remap_ntok");
            }
            require_status(backend_.copy_i32_from_host(
                *bs_mtp_remap_to_dev_, 0, bs_mtp_remap_to_host_.data(),
                mtp_inplace_blocks));
            require_status(backend_.copy_i32_from_host(
                *bs_mtp_remap_from_dev_, 0,
                bs_mtp_remap_from_host_.data(), mtp_inplace_blocks));
            require_status(backend_.copy_i32_from_host(
                *bs_mtp_remap_ntok_dev_, 0,
                bs_mtp_remap_ntok_host_.data(), mtp_inplace_blocks));
            require_status(
                backend_.rope_block_remap_paged_batched_device(
                    mtp_k_cache(), mtp_inplace_blocks,
                    max_mtp_block_tokens, cfg.n_kv_heads, per_pos,
                    cfg.head_dim, cfg.rope_dim, *bs_mtp_remap_to_dev_,
                    *bs_mtp_remap_from_dev_, *bs_mtp_remap_ntok_dev_,
                    *mtp_window_pages_device_, page_size, cfg.rope_theta,
                    kvmem_rope_sincos_.get(),
                    kvmem_rope_table_positions_));
        }
        for (const KvMemRemap &rm : plan.remaps) {
            mtp_baked_pos_[rm.block_id] = static_cast<int64_t>(rm.to_base);
        }
    } else if (build_mtp_window && !kvmem_no_rerope_) {
        kvmem_sync_mtp_baked_pos();
        bs_mtp_remap_to_host_.clear();
        bs_mtp_remap_from_host_.clear();
        bs_mtp_remap_ntok_host_.clear();
        uint32_t max_mtp_block_tokens = 0;
        for (const KvMemRemap &rm : plan.remaps) {
            const int64_t mtp_from = mtp_baked_pos_[rm.block_id];
            if (mtp_from == static_cast<int64_t>(rm.to_base)) continue;  // skip
            trace_rope_position_if_out_of_range(
                "kvmem_assemble.mtp.from", mtp_from, rm.n_tokens,
                cfg.n_ctx_train);
            trace_rope_position_if_out_of_range(
                "kvmem_assemble.mtp.to", rm.to_base, rm.n_tokens,
                cfg.n_ctx_train);
            bs_mtp_remap_to_host_.push_back(rm.to_base);
            bs_mtp_remap_from_host_.push_back(static_cast<int32_t>(mtp_from));
            bs_mtp_remap_ntok_host_.push_back(static_cast<int32_t>(rm.n_tokens));
            max_mtp_block_tokens = std::max(max_mtp_block_tokens, rm.n_tokens);
        }
        const uint32_t n_mtp_moved =
            static_cast<uint32_t>(bs_mtp_remap_to_host_.size());
        if (n_mtp_moved > 0) {
            if (n_mtp_moved > bs_mtp_remap_capacity_) {
                bs_mtp_remap_capacity_ = n_mtp_moved;
                bs_mtp_remap_to_dev_ =
                    backend_.tensor_i32(bs_mtp_remap_capacity_, "bs_mtp_remap_to");
                bs_mtp_remap_from_dev_ =
                    backend_.tensor_i32(bs_mtp_remap_capacity_, "bs_mtp_remap_from");
                bs_mtp_remap_ntok_dev_ =
                    backend_.tensor_i32(bs_mtp_remap_capacity_, "bs_mtp_remap_ntok");
            }
            require_status(backend_.copy_i32_from_host(
                *bs_mtp_remap_to_dev_, 0, bs_mtp_remap_to_host_.data(), n_mtp_moved));
            require_status(backend_.copy_i32_from_host(
                *bs_mtp_remap_from_dev_, 0, bs_mtp_remap_from_host_.data(),
                n_mtp_moved));
            require_status(backend_.copy_i32_from_host(
                *bs_mtp_remap_ntok_dev_, 0, bs_mtp_remap_ntok_host_.data(),
                n_mtp_moved));
            require_status(backend_.rope_block_remap_paged_batched_device(
                mtp_k_cache(), n_mtp_moved, max_mtp_block_tokens, cfg.n_kv_heads,
                per_pos, cfg.head_dim, cfg.rope_dim, *bs_mtp_remap_to_dev_,
                *bs_mtp_remap_from_dev_, *bs_mtp_remap_ntok_dev_,
                *mtp_window_pages_device_, page_size, cfg.rope_theta));
        }
        // Commit the new MTP bake for EVERY selected block (skipped blocks are
        // already at their to_base, so this is a no-op for them).
        for (const KvMemRemap &rm : plan.remaps) {
            mtp_baked_pos_[rm.block_id] = static_cast<int64_t>(rm.to_base);
        }
    }
    uint64_t t_rerope = 0;
    if (tm) {
        require_status(backend_.synchronize());
        t_rerope = kvmem_steady_ns();
    }
    kvmem_active_ = true;
    // Rebuild the per-block representative K + reset the score accumulator for
    // the new interval (k̄ is read from the just-re-RoPE'd window K).
    kvmem_recompute_kbar();
    if (tm) {
        require_status(backend_.synchronize());
        const uint64_t t_end = kvmem_steady_ns();
        KvMemTimingTotals &t = kvmem_timing_totals();
        t.assemble_pages_ns.fetch_add(t_pages - t_asm0,
                                      std::memory_order_relaxed);
        t.assemble_rerope_ns.fetch_add(t_rerope - t_pages,
                                       std::memory_order_relaxed);
        t.assemble_kbar_ns.fetch_add(t_end - t_rerope,
                                     std::memory_order_relaxed);
        t.assemble_ns.fetch_add(t_end - t_asm0, std::memory_order_relaxed);
        t.assemble_calls.fetch_add(1, std::memory_order_relaxed);
        if (kvmem_perf_trace_flag()) {
            std::fprintf(
                stderr,
                "[kvmem-assembly-perf] mode=%s total_ms=%.3f "
                "pages_ms=%.3f rerope_ms=%.3f kbar_ms=%.3f "
                "raw_gather_ms=%.3f raw_h2d_submit_ms=%.3f "
                "raw_h2d_wait_ms=%.3f raw_bytes=%llu raw_batches=%u "
                "raw_refresh_blocks=%zu inplace_blocks=%u "
                "mtp_blocks=%u mtp_inplace_blocks=%u\n",
                kvmem_raw_pipeline_enabled_
                    ? "pipeline"
                    : (kvmem_rope_sincos_ ? "table" : "legacy"),
                (t_end - t_asm0) / 1.0e6,
                (t_pages - t_asm0) / 1.0e6,
                (t_rerope - t_pages) / 1.0e6,
                (t_end - t_rerope) / 1.0e6,
                kvmem_assembly_raw_gather_ns_ / 1.0e6,
                kvmem_assembly_raw_h2d_submit_ns_ / 1.0e6,
                kvmem_assembly_raw_h2d_wait_ns_ / 1.0e6,
                static_cast<unsigned long long>(
                    kvmem_assembly_raw_bytes_),
                kvmem_assembly_raw_batches_, raw_refreshes.size(), n_moved,
                mtp_rebuild_blocks, mtp_inplace_blocks);
        }
    }
}

uint32_t QwenExecutor::kvmem_reselect() {
    (void)kvmem_prepare_reselect();
    return kvmem_finish_reselect();
}

void QwenExecutor::kvmem_prepare_content_index_before_first_selection() {
    if (!block_store_) return;
    const KvMemMethod method = block_store_->config().select_method;
    if (method == KvMemMethod::Retrieval && !g_content_ready_ &&
        !kvmem_active_ && kvmem_query_end_ <= kvmem_query_begin_) {
        kvmem_build_content_index();
    }
}

uint32_t QwenExecutor::kvmem_reselect_prefill_pressure() {
    if (!kvmem_enabled_ || !block_store_) return 0;
    if (block_store_->block_count() == 0) return 0;
    if (kvmem_pending_reselect_) {
        throw std::runtime_error(
            "KVMem prefill-pressure selection during pending reselect");
    }

    // Non-query-conditioned retrieval historically snapshots its immutable
    // content index immediately before the first mid-prefill eviction. Keep
    // that correctness side effect, but do not score or consult it here.
    kvmem_prepare_content_index_before_first_selection();
    const std::vector<uint32_t> selected =
        block_store_->pick_prefill_pressure_blocks();
    if (std::getenv("QW3_KVMEM_TRACE")) {
        const uint32_t n = block_store_->block_count();
        const uint32_t sink = std::min<uint32_t>(
            block_store_->config().sink_blocks,
            static_cast<uint32_t>(selected.size()));
        const uint32_t tail_begin = selected.size() > sink
            ? selected[sink] : n;
        std::fprintf(stderr,
                     "[bs-prefill-pressure] blocks=%u selected=%zu sink=%u "
                     "tail_begin=%u\n",
                     n, selected.size(), sink, tail_begin);
    }
    return kvmem_set_selection(selected);
}

void QwenExecutor::kvmem_prepare_prefill_window(uint32_t upcoming_tokens) {
    if (!kvmem_enabled_ || !block_store_ || !kvmem_active_) return;
    if (upcoming_tokens == 0) return;
    if (kvmem_keep_selected_prefill_) return;
    const uint64_t projected_tokens =
        static_cast<uint64_t>(position_) + upcoming_tokens;
    if (projected_tokens <= block_store_->config().select_budget) return;
    (void)kvmem_reselect_prefill_pressure();
}

uint32_t QwenExecutor::kvmem_prepare_reselect() {
    if (!kvmem_enabled_ || !block_store_) return 0;
    if (block_store_->block_count() == 0) return 0;
    if (kvmem_pending_reselect_) {
        throw std::runtime_error("KVMem reselect already prepared");
    }
    const bool had_query_prefetch = kvmem_query_prefetch_active_;
    const uint64_t query_prefetch_start_ns =
        kvmem_query_prefetch_start_ns_;
    if (had_query_prefetch) {
        try {
            kvmem_finish_prefetch();
        } catch (...) {
            kvmem_query_prefetch_active_ = false;
            kvmem_query_prefetch_blocks_.clear();
            kvmem_query_prefetch_start_ns_ = 0;
            throw;
        }
        kvmem_query_prefetch_active_ = false;
    }
    const bool perf_trace = kvmem_perf_trace_flag();
    if (perf_trace) {
        static std::atomic<uint64_t> reselect_sequence{0};
        kvmem_reselect_perf_ = KvMemReselectPerf{};
        kvmem_reselect_perf_.active = true;
        kvmem_reselect_perf_.sequence =
            reselect_sequence.fetch_add(1, std::memory_order_relaxed);
        kvmem_reselect_perf_.start_ns = kvmem_steady_ns();
    }
    if (kvmem_immutable_source_k_) kvmem_flush_raw_k_decode();
    const bool tm = kvmem_measure_timing_flag();
    const uint64_t t_sel0 = tm ? kvmem_steady_ns() : 0;
    const KvMemMethod method = block_store_->config().select_method;
    // Build the global content-frame index once, from the pristine post-prefill
    // cache (every block still baked at its true position). After the first
    // assembly re-RoPEs selected blocks into window slots this is no longer
    // possible, but the content mean is position-invariant so one build serves
    // the whole session. Only needed for the Retrieval method.
    //
    // When a query-conditioned span is active the index is instead built
    // incrementally during prefill (kvmem_capture_kbar_multi), covering EVERY
    // block — so skip the one-shot paged builder, which can only cover the blocks
    // resident at the first mid-prefill offload (#91). The incremental path sets
    // g_content_ready_ on completion, so this guard is also self-disabling once it
    // has run; the explicit span check keeps the paged builder off even on the
    // transient mid-prefill reselects (before the incremental index completes).
    kvmem_prepare_content_index_before_first_selection();
    // Score selection by the configured method (all three feed pick_topk, which
    // always keeps the sink + recent windows):
    //   Retrieval — global content similarity (can resurrect dropped blocks);
    //               falls back to the window-local heat fold when the
    //               index/query isn't live (q8, or first selection). FP8 has
    //               a typed mean-index and scoring path.
    //   H2O       — window-local cumulative attention heat only.
    //   Recency   — no learned signal; pick_topk keeps sink + recent only.
    const char *scorer_requested = "none";
    const char *scorer_used = "none";
    std::string scorer_fallback_reason;
    auto add_scorer_failure = [&](const char *scorer,
                                  const std::string &reason) {
        if (!scorer_fallback_reason.empty()) {
            scorer_fallback_reason.push_back(',');
        }
        scorer_fallback_reason += scorer;
        scorer_fallback_reason.push_back(':');
        scorer_fallback_reason += reason.empty() ? "unknown_failure" : reason;
    };
    switch (method) {
        case KvMemMethod::Retrieval:
            // Preserve the just-finished step's window-local profile before
            // the global retrieval scorer overwrites the ranking score. Quota
            // selection can then draw from both pools.
            kvmem_drain_scores();
            // Query-conditioned: when a question span was captured during prefill,
            // rank blocks with the CLI-selected scorer (--kvmem-retrieval-method):
            //   per-token -> raw-key ExactMass (global per-token softmax mass);
            //   mean-k    -> softmax-over-pages on the per-block mean key (default).
            // Either falls back to the single last-token content scorer if its
            // buffer isn't live (e.g. no query span, or the shmem page cap is hit).
            if (kvmem_qc_deltanet_) {
                // DeltaNet-state retrieval (deltanet_retrieval.md). Falls back to
                // mean-k / single-token content scoring if its capture isn't live.
                scorer_requested = "deltanet";
                std::string reason;
                bool scored = kvmem_retrieval_score_deltanet(&reason);
                if (scored) scorer_used = "deltanet";
                if (!scored) {
                    add_scorer_failure("deltanet", reason);
                    reason.clear();
                    scored = kvmem_retrieval_score_mean_softmax(-1, &reason);
                    if (scored) {
                        scorer_used = "mean-k";
                    } else {
                        add_scorer_failure("mean-k", reason);
                    }
                }
                if (!scored) {
                    reason.clear();
                    scored = kvmem_retrieval_score(&reason);
                    if (scored) {
                        scorer_used = "single-token-content";
                    } else {
                        add_scorer_failure("single-token-content", reason);
                        scorer_used = "window-profile";
                    }
                }
            } else {
                scorer_requested = kvmem_qc_pertoken_ ? "per-token" : "mean-k";
                std::string reason;
                bool scored = kvmem_qc_pertoken_
                    ? kvmem_retrieval_score_exactmass(&reason)
                    : kvmem_retrieval_score_mean_softmax(-1, &reason);
                if (scored) {
                    scorer_used = scorer_requested;
                } else {
                    add_scorer_failure(scorer_requested, reason);
                    reason.clear();
                    scored = kvmem_retrieval_score(&reason);
                    if (scored) {
                        scorer_used = "single-token-content";
                    } else {
                        add_scorer_failure("single-token-content", reason);
                        scorer_used = "window-profile";
                    }
                }
            }
            break;
        case KvMemMethod::H2O:
            kvmem_drain_scores();
            break;
        case KvMemMethod::Recency:
            // Drop any heat the per-step accumulator gathered so it does not
            // leak into pick_topk; selection stays sink + recent only.
            bs_score_ready_ = false;
            break;
    }
    kvmem_pending_plan_ =
        block_store_->set_selection(kvmem_selection_with_pin());
    if (perf_trace || std::getenv("QW3_KVMEM_TRACE")) {
        std::fprintf(
            stderr,
            "[kvmem-reuse] kind=semantic enabled=%d "
            "selection_overlap_blocks=%u gpu_reused_blocks=%u "
            "retained_position_stable=%u retained_position_moved=%u "
            "stage_in_blocks=%zu stage_out_blocks=%zu\n",
            block_store_->config().hierarchical_reuse ? 1 : 0,
            kvmem_pending_plan_.selection_overlap_blocks,
            kvmem_pending_plan_.gpu_reused_blocks,
            kvmem_pending_plan_.retained_position_stable,
            kvmem_pending_plan_.retained_position_moved,
            kvmem_pending_plan_.stage_in.size(),
            kvmem_pending_plan_.stage_out.size());
    }
    if (had_query_prefetch) {
        uint32_t hits = 0;
        for (uint32_t id : kvmem_query_prefetch_blocks_) {
            if (id < block_store_->block_count() &&
                block_store_->blocks()[id].in_working_set) {
                ++hits;
            }
        }
        if (perf_trace || std::getenv("QW3_KVMEM_TRACE")) {
            const double elapsed_ms =
                query_prefetch_start_ns > 0
                    ? (kvmem_steady_ns() - query_prefetch_start_ns) / 1.0e6
                    : 0.0;
            std::fprintf(
                stderr,
                "[kvmem-query-prefetch] phase=finish candidates=%zu "
                "hits=%u hit_rate=%.6f elapsed_ms=%.3f\n",
                kvmem_query_prefetch_blocks_.size(), hits,
                kvmem_query_prefetch_blocks_.empty()
                    ? 0.0
                    : static_cast<double>(hits) /
                          kvmem_query_prefetch_blocks_.size(),
                elapsed_ms);
        }
        kvmem_query_prefetch_blocks_.clear();
        kvmem_query_prefetch_start_ns_ = 0;
    }
    const KvMemStoreConfig &store_cfg = block_store_->config();
    const bool heat_aware_cpu_cache =
        store_cfg.legacy_optimization_profile
            ? store_cfg.optimization_level >= KvMemOptimizationLevel::Opt1
            : store_cfg.hierarchical_reuse;
    if (kvmem_cpu_tier_ && heat_aware_cpu_cache) {
        kvmem_cpu_tier_->begin_selection_epoch();
        for (const KvMemRemap &rm : kvmem_pending_plan_.remaps) {
            kvmem_cpu_tier_->record_selected(rm.block_id);
        }
    }
    if (method == KvMemMethod::Retrieval) {
        const bool fallback =
            std::strcmp(scorer_requested, scorer_used) != 0;
        if (fallback || std::getenv("QW3_KVMEM_TRACE")) {
            static std::atomic<uint64_t> scorer_event_sequence{0};
            const uint64_t seq =
                scorer_event_sequence.fetch_add(1, std::memory_order_relaxed);
            uint32_t selected_blocks = 0;
            for (const KvMemBlock &block : block_store_->blocks()) {
                selected_blocks += block.in_working_set ? 1u : 0u;
            }
            const uint32_t query_expected =
                kvmem_query_end_ > kvmem_query_begin_
                    ? kvmem_query_end_ - kvmem_query_begin_
                    : 0;
            uint32_t indexed_blocks = 0;
            if (std::strcmp(scorer_used, "per-token") == 0 ||
                std::strcmp(scorer_used, "deltanet") == 0) {
                indexed_blocks = kvmem_qc_total_blocks_;
            } else if (std::strcmp(scorer_used, "mean-k") == 0) {
                indexed_blocks = g_kbar_multi_blocks_;
            } else if (std::strcmp(scorer_used,
                                   "single-token-content") == 0) {
                indexed_blocks = g_indexed_blocks_;
            }
            std::fprintf(
                stderr,
                "[kvmem-scorer] seq=%llu tag=%s position=%u "
                "requested=%s used=%s fallback=%d reason=%s "
                "query_expected=%u query_captured=%u query_ready=%d "
                "indexed_blocks=%u source_blocks=%u selected_blocks=%u\n",
                static_cast<unsigned long long>(seq),
                kvmem_trace_tag_.empty() ? "-" : kvmem_trace_tag_.c_str(),
                position_, scorer_requested, scorer_used, fallback ? 1 : 0,
                scorer_fallback_reason.empty()
                    ? "none"
                    : scorer_fallback_reason.c_str(),
                query_expected, g_query_multi_count_,
                g_query_multi_ready_ ? 1 : 0, indexed_blocks,
                block_store_->block_count(), selected_blocks);
        }
    }
    // DIAGNOSTIC (default OFF): dump the per-block retrieval ranking + selection
    // so an offline tool can locate evidence and compare selected history spans.
    // Gated on QW3_KVMEM_DUMP_SCORES (a file path); normal runs are unchanged.
    if (const char *dump_path = std::getenv("QW3_KVMEM_DUMP_SCORES")) {
        // Dump once per distinct question span (the post-prefill retrieval), so a
        // single server launch handling several requests emits one snapshot each.
        static int kvmem_dump_seq = 0;
        static uint32_t kvmem_dump_last_qb = 0xffffffffu;
        static uint32_t kvmem_dump_last_qe = 0xffffffffu;
        static std::string kvmem_dump_last_tag;
        // Only dump once the content index the scorer actually uses is LIVE. The
        // warm-checkpoint boundary split calls reselect mid-prefill (with the same
        // question span but only ckpt_split blocks registered and the incremental
        // index still filling), which would otherwise be captured first and latch
        // out the real, full-coverage final reselect. Gating on index-ready skips
        // those intermediate reselects and captures the post-prefill selection.
        const bool index_ready = kvmem_qc_deltanet_
            ? kvmem_dn_ready_
            : (kvmem_qc_pertoken_ ? g_kraw_multi_ready_ : g_kbar_multi_ready_);
        const bool request_is_new = !kvmem_trace_tag_.empty()
            ? kvmem_trace_tag_ != kvmem_dump_last_tag
            : (kvmem_query_begin_ != kvmem_dump_last_qb ||
               kvmem_query_end_ != kvmem_dump_last_qe);
        // Transcript replay can execute thousands of re-selections for one
        // request. Dump only its final answer-producing selection; ordinary
        // one-shot requests keep the existing once-per-tag behavior.
        const bool final_trace_event = kvmem_trace_event_count_ == 0 ||
            kvmem_trace_event_index_ + 1 == kvmem_trace_event_count_;
        const bool span_is_new = index_ready && final_trace_event &&
            (kvmem_query_end_ > kvmem_query_begin_) && request_is_new;
        if (span_is_new) {
            kvmem_dump_last_qb = kvmem_query_begin_;
            kvmem_dump_last_qe = kvmem_query_end_;
            kvmem_dump_last_tag = kvmem_trace_tag_;
            // Write one snapshot (meta + per-block rows) from the CURRENT block
            // store state; `mask_label` tags which softmax-mask config produced
            // the scores (-1 = as-run, 0 = forced no-mask, 1 = forced mask).
            auto dump_snapshot = [&](int mask_label) {
                const auto &cfg = block_store_->config();
                const auto &blks = block_store_->blocks();
                if (std::FILE *f = std::fopen(dump_path, "a")) {
                    uint32_t nsel = 0;
                    for (const auto &b : blks) nsel += b.in_working_set ? 1u : 0u;
                    std::fprintf(f,
                        "{\"type\":\"meta\",\"schema_version\":"
                        "\"qw3_kvmem_retrieval_dump.v2\",\"seq\":%d,"
                        "\"trace_tag\":\"%s\",\"block_count\":%zu,"
                        "\"selected\":%u,\"query_begin\":%u,\"query_end\":%u,"
                        "\"context_begin\":%u,\"context_end\":%u,"
                        "\"prompt_tokens\":%zu,"
                        "\"budget_blocks\":%u,\"recent\":%u,\"sink\":%u,"
                        "\"block_tokens\":%u,\"method\":\"%s\",\"mask\":%d,"
                        "\"scorer_requested\":\"%s\","
                        "\"scorer_used\":\"%s\","
                        "\"scorer_fallback_reason\":\"%s\","
                        "\"query_expected\":%u,\"query_captured\":%u,"
                        "\"qc_total_blocks\":%u,\"qc_captured_blocks\":%u,"
                        "\"trace_event\":%u,\"trace_events\":%u,"
                        "\"index_ready\":%d,\"query_ready\":%d,"
                        "\"immutable_source_k\":%d}\n",
                        kvmem_dump_seq, kvmem_trace_tag_.c_str(), blks.size(), nsel,
                        kvmem_query_begin_, kvmem_query_end_,
                        kvmem_context_begin_, kvmem_context_end_,
                        kvmem_trace_prompt_tokens_.size(),
                        block_store_->budget_blocks(), cfg.recent_blocks,
                        cfg.sink_blocks, cfg.block_tokens,
                        kvmem_qc_deltanet_ ? "deltanet" :
                            (kvmem_qc_pertoken_ ? "per-token" : "mean-k"),
                        mask_label,
                        scorer_requested, scorer_used,
                        scorer_fallback_reason.empty()
                            ? "none"
                            : scorer_fallback_reason.c_str(),
                        kvmem_query_end_ > kvmem_query_begin_
                            ? kvmem_query_end_ - kvmem_query_begin_
                            : 0,
                        g_query_multi_count_,
                        kvmem_qc_total_blocks_, kvmem_qc_captured_blocks_,
                        kvmem_trace_event_index_, kvmem_trace_event_count_,
                        index_ready ? 1 : 0,
                        kvmem_qc_deltanet_ ? (kvmem_dn_ready_ ? 1 : 0)
                                           : (g_query_multi_ready_ ? 1 : 0),
                        cfg.immutable_source_k ? 1 : 0);
                    for (const auto &b : blks) {
                        std::fprintf(f,
                            "{\"b\":%u,\"p0\":%u,\"nt\":%u,\"rs\":%.9g,\"ps\":%.9g,"
                            "\"as\":%.9g,\"sel\":%d,\"tier\":%d,"
                            "\"bp\":%lld,\"mv\":%u",
                            b.block_id, b.orig_pos_start, b.n_tokens,
                            b.retrieval_score, b.profile_score, b.attn_score,
                            b.in_working_set ? 1 : 0, static_cast<int>(b.tier),
                            static_cast<long long>(b.baked_pos),
                            b.remap_count);
                        // For tagged overlap experiments, include exact token ids
                        // only on selected rows. This makes text/coordinate
                        // reconstruction exact without multiplying dump size by
                        // every unselected block.
                        const uint64_t p1 =
                            static_cast<uint64_t>(b.orig_pos_start) + b.n_tokens;
                        if (b.in_working_set && !kvmem_trace_prompt_tokens_.empty() &&
                            p1 <= kvmem_trace_prompt_tokens_.size()) {
                            std::fprintf(f, ",\"tok\":[");
                            for (uint32_t i = 0; i < b.n_tokens; ++i) {
                                if (i != 0) std::fputc(',', f);
                                std::fprintf(f, "%u",
                                    kvmem_trace_prompt_tokens_[b.orig_pos_start + i]);
                            }
                            std::fputc(']', f);
                        }
                        std::fprintf(f, "}\n");
                    }
                    std::fclose(f);
                }
                ++kvmem_dump_seq;
            };
            // Mask sweep (QW3_KVMEM_DUMP_MASK_SWEEP): re-score the SAME prefilled
            // index/query twice (mask off, mask on) and dump each, so a single
            // ~12-min prefill yields an apples-to-apples answer-block ranking with
            // and without kept-band masking. Only the mean-k path supports the
            // override; per-token dumps a single as-run snapshot. Restores the
            // production (env-honoring) selection afterward so downstream stage
            // in/out is unchanged.
            const bool sweep = std::getenv("QW3_KVMEM_DUMP_MASK_SWEEP") != nullptr;
            const bool can_sweep = sweep && !kvmem_qc_pertoken_ &&
                (kvmem_query_end_ > kvmem_query_begin_) && g_query_multi_ready_;
            if (can_sweep) {
                if (kvmem_retrieval_score_mean_softmax(0)) {
                    kvmem_pending_plan_ =
                        block_store_->set_selection(kvmem_selection_with_pin());
                    dump_snapshot(0);
                }
                if (kvmem_retrieval_score_mean_softmax(1)) {
                    kvmem_pending_plan_ =
                        block_store_->set_selection(kvmem_selection_with_pin());
                    dump_snapshot(1);
                }
                // Restore production-consistent scoring/selection.
                (void)kvmem_retrieval_score_mean_softmax(-1);
                kvmem_pending_plan_ =
                    block_store_->set_selection(kvmem_selection_with_pin());
            } else {
                dump_snapshot(-1);
            }
        }
    }
    if (tm) {
        require_status(backend_.synchronize());
        const uint64_t t_sel1 = kvmem_steady_ns();
        KvMemTimingTotals &t = kvmem_timing_totals();
        t.retrieval_ns.fetch_add(t_sel1 - t_sel0,
                                 std::memory_order_relaxed);
        t.retrieval_calls.fetch_add(1, std::memory_order_relaxed);
        if (perf_trace) {
            kvmem_reselect_perf_.selection_ns = t_sel1 - t_sel0;
        }
    }
    // Free the evicted blocks' GPU pages BEFORE staging in the resurrected ones.
    // A query-conditioned reselect can swap a large fraction of the working set
    // in one shot (it resurrects scattered, semantically-relevant blocks rather
    // than re-picking the already-resident recent window). stage_out used to run
    // only in finish_reselect, AFTER start_prefetch had already allocated pages
    // for the stage-ins; with many resurrections that overflowed the bounded GPU
    // page pool (the prefill left it near-full). Evicting first reclaims pages so
    // the stage-in fits: stage_out = resident-minus-window, remaps = window, so
    // the two sets are disjoint and the evicted pages are never needed by the
    // stage-in. The post-eviction resident set is at most the window (<= budget
    // blocks <= pool), so the allocation always fits.
    auto trace_pool = [&](const char *phase) {
        if (!kvmem_tier_trace_enabled() || !kvmem_gpu_page_pool_) return;
        std::fprintf(
            stderr,
            "[kvmem-tier] reselect_pool phase=%s remaps=%zu stage_out=%zu "
            "main_used=%u main_free=%u mtp_used=%u mtp_free=%u\n",
            phase, kvmem_pending_plan_.remaps.size(),
            kvmem_pending_plan_.stage_out.size(),
            kvmem_gpu_page_pool_->used_pages(),
            kvmem_gpu_page_pool_->free_pages(),
            kvmem_mtp_gpu_page_pool_
                ? kvmem_mtp_gpu_page_pool_->used_pages() : 0,
            kvmem_mtp_gpu_page_pool_
                ? kvmem_mtp_gpu_page_pool_->free_pages() : 0);
    };
    trace_pool("before_stage_out");
    kvmem_stage_out(kvmem_pending_plan_.stage_out);
    if (perf_trace) {
        kvmem_reselect_perf_.stage_out_ns =
            kvmem_last_stage_out_perf_.total_ns;
    }
    trace_pool("after_stage_out");
    kvmem_start_prefetch(kvmem_pending_plan_);
    trace_pool("after_start_prefetch");
    kvmem_pending_reselect_ = true;
    return kvmem_pending_plan_.total_window_tokens;
}

uint32_t QwenExecutor::kvmem_finish_reselect() {
    if (!kvmem_pending_reselect_) return window_query_pos_;
    const bool perf_trace =
        kvmem_perf_trace_flag() && kvmem_reselect_perf_.active;
    const bool overlap_stagein_assembly =
        kvmem_stagein_assembly_overlap_enabled_ &&
        kvmem_raw_pipeline_enabled_ &&
        kvmem_prefetch_.active &&
        kvmem_prefetch_.bulk_cpu &&
        !kvmem_prefetch_.cpu_reads.empty() &&
        kvmem_prefetch_.nvme_reads.empty();
    const uint64_t overlap_wall0 =
        perf_trace && overlap_stagein_assembly
            ? kvmem_steady_ns() : 0;
    uint64_t t_assemble0 = 0;
    uint64_t t_assemble_done = 0;
    if (overlap_stagein_assembly) {
        kvmem_stagein_assembly_overlap_active_ = true;
        std::future<void> stagein_future;
        try {
            stagein_future = std::async(
                std::launch::async,
                [this]() { kvmem_finish_prefetch(); });
        } catch (...) {
            kvmem_stagein_assembly_overlap_active_ = false;
            throw;
        }
        std::exception_ptr assemble_error;
        std::exception_ptr stagein_error;
        try {
            t_assemble0 = perf_trace ? kvmem_steady_ns() : 0;
            kvmem_assemble(kvmem_pending_plan_);
            t_assemble_done = perf_trace ? kvmem_steady_ns() : 0;
        } catch (...) {
            assemble_error = std::current_exception();
            t_assemble_done = perf_trace ? kvmem_steady_ns() : 0;
        }
        try {
            stagein_future.get();
        } catch (...) {
            stagein_error = std::current_exception();
        }
        kvmem_stagein_assembly_overlap_active_ = false;
        if (stagein_error) std::rethrow_exception(stagein_error);
        if (assemble_error) std::rethrow_exception(assemble_error);
    } else {
        kvmem_finish_prefetch();
        t_assemble0 = perf_trace ? kvmem_steady_ns() : 0;
        kvmem_assemble(kvmem_pending_plan_);
        t_assemble_done = perf_trace ? kvmem_steady_ns() : 0;
    }
    const uint64_t t_done =
        perf_trace ? kvmem_steady_ns() : 0;
    if (perf_trace) {
        const KvMemPrefetchPerf &in = kvmem_last_prefetch_perf_;
        const KvMemStageOutPerf &out = kvmem_last_stage_out_perf_;
        const double ns_to_ms = 1.0 / 1.0e6;
        const double bytes_to_gib =
            1.0 / (1024.0 * 1024.0 * 1024.0);
        const uint64_t submit_ns =
            in.start_exit_ns >= in.start_enter_ns
                ? in.start_exit_ns - in.start_enter_ns : 0;
        const uint64_t overlap_gap_ns =
            in.finish_enter_ns >= in.start_exit_ns
                ? in.finish_enter_ns - in.start_exit_ns : 0;
        const uint64_t finish_ns =
            in.finish_exit_ns >= in.finish_enter_ns
                ? in.finish_exit_ns - in.finish_enter_ns : 0;
        const uint64_t stage_in_wall_ns =
            in.finish_exit_ns >= in.start_enter_ns
                ? in.finish_exit_ns - in.start_enter_ns : 0;
        const uint64_t nvme_hidden_ns =
            in.nvme_read_ns > in.nvme_wait_ns
                ? in.nvme_read_ns - in.nvme_wait_ns : 0;
        const uint64_t total_ns =
            t_done >= kvmem_reselect_perf_.start_ns
                ? t_done - kvmem_reselect_perf_.start_ns : 0;
        std::fprintf(
            stderr,
            "[kvmem-reselect-perf] kind=semantic seq=%llu tag=%s position=%u "
            "source_blocks=%u selected_blocks=%zu remaps=%zu stage_out=%zu "
            "selection_ms=%.3f stage_out_ms=%.3f "
            "stage_out_d2h_ms=%.3f stage_out_cpu_copy_ms=%.3f "
            "stage_out_disk_write_ms=%.3f stage_out_blocks=%u "
            "stage_out_gib=%.3f stage_out_clean_blocks=%u "
            "stage_out_clean_avoided_gib=%.3f "
            "stage_out_async_submit_ms=%.3f "
            "stage_out_async_gather_ms=%.3f "
            "stage_out_async_backpressure_ms=%.3f "
            "stage_out_async_batches=%u stage_out_async_gib=%.3f "
            "pending_write_batches=%zu async_completed_write_ms=%.3f "
            "async_completed_write_gib=%.3f "
            "async_completed_write_syscalls=%llu "
            "prefetch_submit_ms=%.3f overlap_gap_ms=%.3f "
            "prefetch_finish_ms=%.3f stage_in_wall_ms=%.3f "
            "pending_write_wait_ms=%.3f "
            "cpu_gather_ms=%.3f cpu_h2d_enqueue_ms=%.3f "
            "cpu_h2d_wait_ms=%.3f cpu_h2d_batches=%u "
            "nvme_read_ms=%.3f "
            "nvme_wait_ms=%.3f nvme_hidden_ms=%.3f "
            "nvme_read_batches=%u nvme_read_syscalls=%llu "
            "nvme_h2d_enqueue_ms=%.3f h2d_wait_ms=%.3f "
            "cpu_in_blocks=%u cpu_in_gib=%.3f "
            "nvme_in_blocks=%u nvme_in_gib=%.3f "
            "stagein_assembly_overlap=%d overlap_wall_ms=%.3f "
            "assemble_ms=%.3f total_ms=%.3f\n",
            static_cast<unsigned long long>(
                kvmem_reselect_perf_.sequence),
            kvmem_trace_tag_.empty() ? "-" : kvmem_trace_tag_.c_str(),
            position_, block_store_->block_count(),
            kvmem_pending_plan_.remaps.size(),
            kvmem_pending_plan_.remaps.size(),
            kvmem_pending_plan_.stage_out.size(),
            kvmem_reselect_perf_.selection_ns * ns_to_ms,
            kvmem_reselect_perf_.stage_out_ns * ns_to_ms,
            out.canonicalize_and_d2h_ns * ns_to_ms,
            out.cpu_copy_ns * ns_to_ms,
            out.nvme_write_ns * ns_to_ms, out.blocks,
            out.bytes * bytes_to_gib, out.clean_blocks,
            out.clean_bytes_avoided * bytes_to_gib,
            out.async_submit_ns * ns_to_ms,
            out.async_gather_ns * ns_to_ms,
            out.async_backpressure_ns * ns_to_ms,
            out.async_batches,
            out.async_submitted_bytes * bytes_to_gib,
            kvmem_pending_writes_.size(),
            kvmem_async_write_completed_ns_ * ns_to_ms,
            kvmem_async_write_completed_bytes_ * bytes_to_gib,
            static_cast<unsigned long long>(
                kvmem_async_write_completed_syscalls_),
            submit_ns * ns_to_ms,
            overlap_gap_ns * ns_to_ms, finish_ns * ns_to_ms,
            stage_in_wall_ns * ns_to_ms,
            in.pending_write_wait_ns * ns_to_ms,
            in.cpu_gather_ns * ns_to_ms,
            in.cpu_h2d_enqueue_ns * ns_to_ms,
            in.cpu_h2d_wait_ns * ns_to_ms,
            in.cpu_h2d_batches,
            in.nvme_read_ns * ns_to_ms, in.nvme_wait_ns * ns_to_ms,
            nvme_hidden_ns * ns_to_ms,
            in.nvme_read_batches,
            static_cast<unsigned long long>(in.nvme_read_syscalls),
            in.nvme_h2d_enqueue_ns * ns_to_ms,
            in.h2d_wait_ns * ns_to_ms, in.cpu_blocks,
            in.cpu_bytes * bytes_to_gib, in.nvme_blocks,
            in.nvme_bytes * bytes_to_gib,
            overlap_stagein_assembly ? 1 : 0,
            overlap_stagein_assembly
                ? (t_done - overlap_wall0) * ns_to_ms
                : 0.0,
            (t_assemble_done - t_assemble0) * ns_to_ms,
            total_ns * ns_to_ms);
        if (kvmem_cpu_tier_) {
            const PinnedKvCacheStats &cache = kvmem_cpu_tier_->stats();
            const size_t incoming = kvmem_pending_plan_.stage_in.size();
            const size_t retained =
                kvmem_pending_plan_.remaps.size() >= incoming
                    ? kvmem_pending_plan_.remaps.size() - incoming
                    : 0;
            const double cpu_hit_rate = incoming > 0
                ? static_cast<double>(in.cpu_blocks) /
                      static_cast<double>(incoming)
                : 1.0;
            std::fprintf(
                stderr,
                "[kvmem-cache] level=%s kind=semantic retained=%zu "
                "incoming=%zu cpu_hits=%u ssd_misses=%u "
                "incoming_cpu_hit_rate=%.6f selection_epochs=%llu "
                "admissions=%llu admission_rejections=%llu evictions=%llu\n",
                kvmem_optimization_level_name(
                    block_store_->config().optimization_level),
                retained, incoming, in.cpu_blocks, in.nvme_blocks,
                cpu_hit_rate,
                static_cast<unsigned long long>(cache.selection_epochs),
                static_cast<unsigned long long>(cache.admissions),
                static_cast<unsigned long long>(
                    cache.admission_rejections),
                static_cast<unsigned long long>(cache.evictions));
        }
    }
    kvmem_pending_reselect_ = false;
    kvmem_pending_plan_ = KvMemPlan{};
    kvmem_reselect_perf_ = KvMemReselectPerf{};
    return window_query_pos_;
}

uint32_t QwenExecutor::kvmem_set_selection(
        const std::vector<uint32_t> &block_ids) {
    if (!kvmem_enabled_ || !block_store_) return 0;
    if (block_store_->block_count() == 0) return 0;
    if (kvmem_pending_reselect_) {
        throw std::runtime_error("KVMem explicit selection during pending reselect");
    }
    const bool perf_trace = kvmem_perf_trace_flag();
    const uint64_t t_all0 =
        perf_trace ? kvmem_steady_ns() : 0;
    if (kvmem_immutable_source_k_) kvmem_flush_raw_k_decode();
    const uint64_t t_select0 =
        perf_trace ? kvmem_steady_ns() : 0;
    KvMemPlan plan = block_store_->set_selection(block_ids);
    const uint64_t selection_ns =
        perf_trace ? kvmem_steady_ns() - t_select0 : 0;
    if (perf_trace || std::getenv("QW3_KVMEM_TRACE")) {
        std::fprintf(
            stderr,
            "[kvmem-reuse] kind=explicit enabled=%d "
            "selection_overlap_blocks=%u gpu_reused_blocks=%u "
            "retained_position_stable=%u retained_position_moved=%u "
            "stage_in_blocks=%zu stage_out_blocks=%zu\n",
            block_store_->config().hierarchical_reuse ? 1 : 0,
            plan.selection_overlap_blocks, plan.gpu_reused_blocks,
            plan.retained_position_stable, plan.retained_position_moved,
            plan.stage_in.size(), plan.stage_out.size());
    }
    // Evict before staging in (same pool-headroom invariant as the deferred
    // prepare/finish path): a large working-set swap would otherwise exhaust the
    // bounded GPU page pool when start_prefetch allocates ahead of stage_out.
    auto trace_pool = [&](const char *phase) {
        if (!kvmem_tier_trace_enabled() || !kvmem_gpu_page_pool_) return;
        std::fprintf(
            stderr,
            "[kvmem-tier] explicit_select_pool phase=%s remaps=%zu "
            "stage_out=%zu main_used=%u main_free=%u mtp_used=%u mtp_free=%u\n",
            phase, plan.remaps.size(), plan.stage_out.size(),
            kvmem_gpu_page_pool_->used_pages(),
            kvmem_gpu_page_pool_->free_pages(),
            kvmem_mtp_gpu_page_pool_
                ? kvmem_mtp_gpu_page_pool_->used_pages() : 0,
            kvmem_mtp_gpu_page_pool_
                ? kvmem_mtp_gpu_page_pool_->free_pages() : 0);
    };
    trace_pool("before_stage_out");
    kvmem_stage_out(plan.stage_out);
    trace_pool("after_stage_out");
    kvmem_start_prefetch(plan);
    trace_pool("after_start_prefetch");
    const bool overlap_stagein_assembly =
        kvmem_stagein_assembly_overlap_enabled_ &&
        kvmem_raw_pipeline_enabled_ &&
        kvmem_prefetch_.active &&
        kvmem_prefetch_.bulk_cpu &&
        !kvmem_prefetch_.cpu_reads.empty() &&
        kvmem_prefetch_.nvme_reads.empty();
    const uint64_t overlap_wall0 =
        perf_trace && overlap_stagein_assembly
            ? kvmem_steady_ns() : 0;
    uint64_t t_assemble0 = 0;
    uint64_t t_assemble_done = 0;
    if (overlap_stagein_assembly) {
        kvmem_stagein_assembly_overlap_active_ = true;
        std::future<void> stagein_future;
        try {
            stagein_future = std::async(
                std::launch::async,
                [this]() { kvmem_finish_prefetch(); });
        } catch (...) {
            kvmem_stagein_assembly_overlap_active_ = false;
            throw;
        }
        std::exception_ptr assemble_error;
        std::exception_ptr stagein_error;
        try {
            t_assemble0 = perf_trace ? kvmem_steady_ns() : 0;
            kvmem_assemble(plan);
            t_assemble_done = perf_trace ? kvmem_steady_ns() : 0;
        } catch (...) {
            assemble_error = std::current_exception();
            t_assemble_done = perf_trace ? kvmem_steady_ns() : 0;
        }
        try {
            stagein_future.get();
        } catch (...) {
            stagein_error = std::current_exception();
        }
        kvmem_stagein_assembly_overlap_active_ = false;
        if (stagein_error) std::rethrow_exception(stagein_error);
        if (assemble_error) std::rethrow_exception(assemble_error);
    } else {
        kvmem_finish_prefetch();
        t_assemble0 = perf_trace ? kvmem_steady_ns() : 0;
        kvmem_assemble(plan);
        t_assemble_done = perf_trace ? kvmem_steady_ns() : 0;
    }
    if (perf_trace) {
        static std::atomic<uint64_t> explicit_sequence{0};
        const uint64_t done = kvmem_steady_ns();
        const KvMemPrefetchPerf &in = kvmem_last_prefetch_perf_;
        const KvMemStageOutPerf &out = kvmem_last_stage_out_perf_;
        const uint64_t submit_ns =
            in.start_exit_ns >= in.start_enter_ns
                ? in.start_exit_ns - in.start_enter_ns : 0;
        const uint64_t overlap_gap_ns =
            in.finish_enter_ns >= in.start_exit_ns
                ? in.finish_enter_ns - in.start_exit_ns : 0;
        const uint64_t finish_ns =
            in.finish_exit_ns >= in.finish_enter_ns
                ? in.finish_exit_ns - in.finish_enter_ns : 0;
        const uint64_t stage_in_wall_ns =
            in.finish_exit_ns >= in.start_enter_ns
                ? in.finish_exit_ns - in.start_enter_ns : 0;
        const uint64_t nvme_hidden_ns =
            in.nvme_read_ns > in.nvme_wait_ns
                ? in.nvme_read_ns - in.nvme_wait_ns : 0;
        constexpr double ns_to_ms = 1.0 / 1.0e6;
        constexpr double bytes_to_gib =
            1.0 / (1024.0 * 1024.0 * 1024.0);
        std::fprintf(
            stderr,
            "[kvmem-reselect-perf] kind=explicit seq=%llu tag=%s "
            "position=%u source_blocks=%u selected_blocks=%zu remaps=%zu "
            "stage_out=%zu selection_ms=%.3f stage_out_ms=%.3f "
            "stage_out_d2h_ms=%.3f stage_out_cpu_copy_ms=%.3f "
            "stage_out_disk_write_ms=%.3f "
            "stage_out_blocks=%u stage_out_gib=%.3f "
            "stage_out_clean_blocks=%u stage_out_clean_avoided_gib=%.3f "
            "stage_out_async_submit_ms=%.3f "
            "stage_out_async_gather_ms=%.3f "
            "stage_out_async_backpressure_ms=%.3f "
            "stage_out_async_batches=%u stage_out_async_gib=%.3f "
            "pending_write_batches=%zu async_completed_write_ms=%.3f "
            "async_completed_write_gib=%.3f "
            "async_completed_write_syscalls=%llu "
            "prefetch_submit_ms=%.3f overlap_gap_ms=%.3f "
            "prefetch_finish_ms=%.3f stage_in_wall_ms=%.3f "
            "pending_write_wait_ms=%.3f "
            "cpu_gather_ms=%.3f cpu_h2d_enqueue_ms=%.3f "
            "cpu_h2d_wait_ms=%.3f cpu_h2d_batches=%u "
            "nvme_read_ms=%.3f nvme_wait_ms=%.3f nvme_hidden_ms=%.3f "
            "nvme_read_batches=%u nvme_read_syscalls=%llu "
            "nvme_h2d_enqueue_ms=%.3f h2d_wait_ms=%.3f "
            "cpu_in_blocks=%u cpu_in_gib=%.3f "
            "nvme_in_blocks=%u nvme_in_gib=%.3f "
            "stagein_assembly_overlap=%d overlap_wall_ms=%.3f "
            "assemble_ms=%.3f total_ms=%.3f\n",
            static_cast<unsigned long long>(
                explicit_sequence.fetch_add(1, std::memory_order_relaxed)),
            kvmem_trace_tag_.empty() ? "-" : kvmem_trace_tag_.c_str(),
            position_, block_store_->block_count(), plan.remaps.size(),
            plan.remaps.size(), plan.stage_out.size(),
            selection_ns * ns_to_ms, out.total_ns * ns_to_ms,
            out.canonicalize_and_d2h_ns * ns_to_ms,
            out.cpu_copy_ns * ns_to_ms,
            out.nvme_write_ns * ns_to_ms, out.blocks,
            out.bytes * bytes_to_gib, out.clean_blocks,
            out.clean_bytes_avoided * bytes_to_gib,
            out.async_submit_ns * ns_to_ms,
            out.async_gather_ns * ns_to_ms,
            out.async_backpressure_ns * ns_to_ms,
            out.async_batches,
            out.async_submitted_bytes * bytes_to_gib,
            kvmem_pending_writes_.size(),
            kvmem_async_write_completed_ns_ * ns_to_ms,
            kvmem_async_write_completed_bytes_ * bytes_to_gib,
            static_cast<unsigned long long>(
                kvmem_async_write_completed_syscalls_),
            submit_ns * ns_to_ms,
            overlap_gap_ns * ns_to_ms, finish_ns * ns_to_ms,
            stage_in_wall_ns * ns_to_ms,
            in.pending_write_wait_ns * ns_to_ms,
            in.cpu_gather_ns * ns_to_ms,
            in.cpu_h2d_enqueue_ns * ns_to_ms,
            in.cpu_h2d_wait_ns * ns_to_ms,
            in.cpu_h2d_batches,
            in.nvme_read_ns * ns_to_ms, in.nvme_wait_ns * ns_to_ms,
            nvme_hidden_ns * ns_to_ms,
            in.nvme_read_batches,
            static_cast<unsigned long long>(in.nvme_read_syscalls),
            in.nvme_h2d_enqueue_ns * ns_to_ms,
            in.h2d_wait_ns * ns_to_ms, in.cpu_blocks,
            in.cpu_bytes * bytes_to_gib, in.nvme_blocks,
            in.nvme_bytes * bytes_to_gib,
            overlap_stagein_assembly ? 1 : 0,
            overlap_stagein_assembly
                ? (done - overlap_wall0) * ns_to_ms
                : 0.0,
            (t_assemble_done - t_assemble0) * ns_to_ms,
            (done - t_all0) * ns_to_ms);
    }
    return window_query_pos_;
}

// Ensure the window page table has a physical page covering window slot
// `window_query_pos_` (the slot the next decode token will occupy). The new
// token's true KV was just allocated in kv_pages_ at logical `position_`; in
// the no-copy design the window's trailing slot aliases that same physical
// page. Under identity selection window_query_pos_ == position_, so this
// appends the exact page kv_pages_ just grew — keeping the two tables in
// lockstep and the decode path byte-identical to plain.
void QwenExecutor::kvmem_extend_window_for_decode() {
    const uint32_t page_size = kv_pages_.page_size;
    const uint32_t need_pages = (window_query_pos_ + page_size) / page_size;
    if (need_pages <= window_page_count_) return;  // current page has room
    // The true tail page kv_pages_ just allocated for `position_`.
    const uint32_t true_page = position_ / page_size;
    if (true_page >= kv_pages_.pages.size()) {
        throw std::runtime_error(
            "block-sparse decode: true KV page not allocated before window extend");
    }
    window_pages_host_.push_back(kv_pages_.pages[true_page]);
    window_page_count_ = static_cast<uint32_t>(window_pages_host_.size());
    sync_window_pages_device(window_page_count_);
}

// Batched analogue of kvmem_extend_window_for_decode: grow the window so `n`
// tokens can be appended starting at window slot window_query_pos_. The true
// tail pages for [position_, position_+n) must already be allocated (caller
// runs ensure_kv_pages first); we alias the SAME physical pages into the window
// page list (no copy — window slots and true slots share pages during decode,
// since window_query_pos_ and position_ advance in lockstep). This pushes the
// exact same physical pages, in the same order, that `n` successive
// kvmem_extend_window_for_decode() calls would (each decode token advances
// position_ and window_query_pos_ together, so window page `pg` aliases true
// page `pg + (position_ - window_query_pos_)/page_size`). Syncs once at the end.
void QwenExecutor::kvmem_extend_window_for_decode_n(uint32_t n,
                                                    uint32_t true_base_pos) {
    if (n == 0) return;
    const uint32_t page_size = kv_pages_.page_size;
    const uint32_t need_pages =
        (window_query_pos_ + n + page_size - 1) / page_size;
    if (need_pages <= window_page_count_) return;  // current pages have room
    // Constant page offset between the window tail and the true cache tail;
    // window_query_pos_ <= true_base_pos always (the window is a compressed view).
    if (true_base_pos < window_query_pos_) {
        throw std::runtime_error(
            "block-sparse batched decode: true base precedes window tail");
    }
    const uint32_t delta = (true_base_pos - window_query_pos_) / page_size;
    for (uint32_t pg = window_page_count_; pg < need_pages; ++pg) {
        const uint32_t true_page = pg + delta;
        if (true_page >= kv_pages_.pages.size()) {
            throw std::runtime_error(
                "block-sparse batched decode: true KV page not allocated "
                "before window extend");
        }
        window_pages_host_.push_back(kv_pages_.pages[true_page]);
    }
    window_page_count_ = static_cast<uint32_t>(window_pages_host_.size());
    sync_window_pages_device(window_page_count_);
}

void QwenExecutor::kvmem_extend_mtp_window_for_decode_n(uint32_t n,
                                                        uint32_t true_base_pos) {
    if (n == 0) return;
    const uint32_t page_size = mtp_kv_pages_.page_size;
    const uint32_t need_pages =
        (window_query_pos_ + n + page_size - 1) / page_size;
    // The host vector is the source of truth for how many window pages actually
    // exist; mtp_window_page_count_ only mirrors it. A state restore can leave
    // the count ahead of the host length (it force-syncs the count to the main
    // window without growing the host), so grow against the host size and
    // re-sync the count from it -- never trust a stale count as the loop base.
    if (mtp_window_pages_host_.size() >= need_pages) {
        mtp_window_page_count_ =
            static_cast<uint32_t>(mtp_window_pages_host_.size());
        return;  // current pages already cover the requested slots
    }
    if (true_base_pos < window_query_pos_) {
        throw std::runtime_error(
            "block-sparse MTP draft: true base precedes window tail");
    }
    const uint32_t delta = (true_base_pos - window_query_pos_) / page_size;
    while (mtp_window_pages_host_.size() < need_pages) {
        const uint32_t pg = static_cast<uint32_t>(mtp_window_pages_host_.size());
        const uint32_t true_page = pg + delta;
        if (true_page >= mtp_kv_pages_.pages.size() ||
            mtp_kv_pages_.pages[true_page] < 0) {
            throw std::runtime_error(
                "block-sparse MTP draft: true MTP KV page not allocated "
                "before window extend");
        }
        mtp_window_pages_host_.push_back(mtp_kv_pages_.pages[true_page]);
    }
    mtp_window_page_count_ = static_cast<uint32_t>(mtp_window_pages_host_.size());
    sync_mtp_window_pages_device(mtp_window_page_count_);
}

// Rebuild the per-window-block representative K (mean baked K) at the first
// standard attention layer and reset the per-block score accumulator. Called at
// the end of assembly, once per reselect interval (cost amortized). Reads the
// just-re-RoPE'd window K so k̄ lives in the same window frame as the Q the
// per-step kernel will score against.
void QwenExecutor::kvmem_recompute_kbar() {
    bs_score_ready_ = false;
    bs_window_blocks_ = static_cast<uint32_t>(bs_window_block_ids_.size());
    if (bs_window_blocks_ == 0) return;

    const QwenConfig &cfg = model_.config();
    // Pick the first standard attention layer as the global representative.
    if (bs_score_layer_ < 0) {
        for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
            if (cfg.is_standard_attention_layer(il)) {
                bs_score_layer_ = static_cast<int32_t>(il);
                break;
            }
        }
    }
    if (bs_score_layer_ < 0) return;  // no standard attention layers (shouldn't happen)

    const uint32_t n_kv_heads = cfg.n_kv_heads;
    const uint32_t head_dim = cfg.head_dim;
    const uint32_t per_pos = n_kv_heads * head_dim;

    // (Re)allocate device buffers when the block capacity grows.
    if (bs_window_blocks_ > bs_kbar_capacity_) {
        bs_kbar_capacity_ = bs_window_blocks_;
        bs_kbar_ = backend_.tensor_f32(
            static_cast<uint64_t>(bs_kbar_capacity_) * n_kv_heads * head_dim,
            "bs_kbar");
        bs_score_accum_ = backend_.tensor_f32(bs_kbar_capacity_, "bs_score_accum");
        bs_win_base_dev_ = backend_.tensor_i32(bs_kbar_capacity_, "bs_win_base");
        bs_blk_tokens_dev_ = backend_.tensor_i32(bs_kbar_capacity_, "bs_blk_tokens");
    }
    require_status(backend_.copy_i32_from_host(
        *bs_win_base_dev_, 0, bs_win_base_host_.data(), bs_window_blocks_));
    require_status(backend_.copy_i32_from_host(
        *bs_blk_tokens_dev_, 0, bs_blk_tokens_host_.data(), bs_window_blocks_));
    require_status(backend_.zero_tensor(*bs_score_accum_));

    // q8's row-scale representation is not supported by the mean builder.
    // fp8 is supported directly (accumulation remains fp32), as are fp16/fp32.
    auto st = backend_.block_kmean_paged_device(
        attention_k_cache(static_cast<uint32_t>(bs_score_layer_)), *bs_kbar_,
        bs_window_blocks_, n_kv_heads, per_pos, head_dim,
        *bs_win_base_dev_, *bs_blk_tokens_dev_, *window_pages_device_,
        kv_pages_.page_size);
    bs_score_ready_ = st.ok;
}

// Per decode step at the representative layer: score the current RoPE-baked Q
// against every window block's k̄ and atomic-add into the GPU-resident
// accumulator. No D2H. Inert unless this is the representative layer and k̄ is
// live for the current interval.
void QwenExecutor::kvmem_score_current_step(uint32_t layer_index,
                                                   float scale) {
    if (!bs_score_ready_ || bs_window_blocks_ == 0) return;
    if (static_cast<int32_t>(layer_index) != bs_score_layer_) return;
    const QwenConfig &cfg = model_.config();
    (void)backend_.block_attn_score_step_device(
        *bs_score_accum_, *q_, *bs_kbar_,
        /*q_stride=*/2 * cfg.head_dim, bs_window_blocks_,
        cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, scale);
}

// Drain the interval's accumulator to host and fold it into the block store's
// cumulative attn_score (indexed by block_id) so the next pick_topk ranks by
// attention heat. Called at the reselect boundary before pick_topk.
void QwenExecutor::kvmem_drain_scores() {
    if (!bs_score_ready_ || bs_window_blocks_ == 0 || !block_store_) return;
    std::vector<float> accum(bs_window_blocks_, 0.0f);
    if (auto st = backend_.copy_to_host(*bs_score_accum_, accum.data(), 0,
                                        bs_window_blocks_);
        !st.ok) {
        bs_score_ready_ = false;
        return;
    }
    std::vector<double> scores(block_store_->block_count(), 0.0);
    for (uint32_t w = 0; w < bs_window_blocks_; ++w) {
        const uint32_t id = bs_window_block_ids_[w];
        if (id < scores.size()) scores[id] += static_cast<double>(accum[w]);
    }
    block_store_->accumulate_attn(scores);
    if (std::getenv("QW3_KVMEM_TRACE")) {
        // Surface the top attention-heat blocks of the interval so the
        // selection signal is observable (internal diagnostic, default off).
        std::vector<std::pair<float, uint32_t>> ranked;
        ranked.reserve(bs_window_blocks_);
        for (uint32_t w = 0; w < bs_window_blocks_; ++w) {
            ranked.emplace_back(accum[w], bs_window_block_ids_[w]);
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto &a, const auto &b) { return a.first > b.first; });
        std::fprintf(stderr, "[bs-trace] interval scores (top): ");
        for (size_t i = 0; i < ranked.size() && i < 6; ++i) {
            std::fprintf(stderr, "blk%u=%.2f ", ranked[i].second, ranked[i].first);
        }
        std::fprintf(stderr, "(window_blocks=%u)\n", bs_window_blocks_);
    }
    bs_score_ready_ = false;
}

bool QwenExecutor::kvmem_attn_trace_enabled() const {
    const char *path = std::getenv("QW3_KVMEM_ATTN_TRACE");
    if (!path || !*path) return false;
    return !env_disabled_value(env_lower_ascii(path));
}

bool QwenExecutor::kvmem_attn_trace_sample_now() const {
    if (!kvmem_active_ || !block_store_ || bs_window_blocks_ == 0) return false;
    if (!kvmem_attn_trace_enabled()) return false;
    const uint32_t interval =
        std::max<uint32_t>(1, env_uint32_or("QW3_KVMEM_ATTN_TRACE_INTERVAL", 1));
    return (kvmem_attn_trace_seen_tokens_ % interval) == 0;
}

void QwenExecutor::kvmem_trace_attention_layer(uint32_t layer_index,
                                               const DeviceTensor &k_cache,
                                               const DeviceTensor &q,
                                               uint32_t q_stride,
                                               const DeviceTensor &page_indices,
                                               uint32_t n_pages,
                                               uint32_t seq_len,
                                               float scale) {
    if (!kvmem_attn_trace_enabled() || !block_store_) return;
    if (bs_window_blocks_ == 0 || !bs_win_base_dev_ || !bs_blk_tokens_dev_) return;

    const uint32_t buckets = bs_window_blocks_ + 1;  // final bucket = decode tail
    if (!kvmem_attn_trace_mass_ || buckets > kvmem_attn_trace_mass_capacity_) {
        kvmem_attn_trace_mass_capacity_ = buckets;
        kvmem_attn_trace_mass_ =
            backend_.tensor_f32(kvmem_attn_trace_mass_capacity_,
                                "kvmem_attn_trace_mass");
    }

    const QwenConfig &cfg = model_.config();
    const uint32_t per_pos = cfg.n_kv_heads * cfg.head_dim;
    auto st = backend_.block_attention_mass_paged_device(
        *kvmem_attn_trace_mass_, q, q_stride, k_cache,
        bs_window_blocks_, cfg.n_heads, cfg.n_kv_heads, per_pos, cfg.head_dim,
        *bs_win_base_dev_, *bs_blk_tokens_dev_, page_indices, n_pages,
        kv_pages_.page_size, seq_len, scale);
    if (!st.ok) return;  // diagnostic-only path: unsupported KV dtype just skips

    std::vector<float> mass(buckets, 0.0f);
    if (auto copy = backend_.copy_to_host(*kvmem_attn_trace_mass_, mass.data(), 0,
                                          buckets);
        !copy.ok) {
        return;
    }

    if (!kvmem_attn_trace_file_) {
        const char *path = std::getenv("QW3_KVMEM_ATTN_TRACE");
        kvmem_attn_trace_file_ = std::fopen(path, "a");
        if (!kvmem_attn_trace_file_) return;
    }

    double sum = 0.0;
    for (float v : mass) sum += static_cast<double>(v);

    std::FILE *f = kvmem_attn_trace_file_;
    std::fprintf(f,
                 "{\"kind\":\"kvmem_attention_mass\","
                 "\"sample\":%llu,\"position\":%u,\"window_query_pos\":%u,"
                 "\"seq_len\":%u,\"layer\":%u,\"n_heads\":%u,"
                 "\"n_kv_heads\":%u,\"head_dim\":%u,\"sum\":%.9g,"
                 "\"block_ids\":[",
                 static_cast<unsigned long long>(kvmem_attn_trace_sample_),
                 position_, window_query_pos_, seq_len, layer_index,
                 cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, sum);
    for (uint32_t i = 0; i < bs_window_blocks_; ++i) {
        if (i) std::fputc(',', f);
        std::fprintf(f, "%u", bs_window_block_ids_[i]);
    }
    if (bs_window_blocks_ > 0) std::fputc(',', f);
    std::fputs("-1],\"mass\":[", f);
    for (uint32_t i = 0; i < buckets; ++i) {
        if (i) std::fputc(',', f);
        std::fprintf(f, "%.9g", static_cast<double>(mass[i]));
    }
    std::fputs("]}\n", f);
    std::fflush(f);
}

bool QwenExecutor::global_attn_trace_enabled() const {
    const char *path = std::getenv("QW3_ATTN_TRACE");
    if (!path || !*path) return false;
    return !env_disabled_value(env_lower_ascii(path));
}

bool QwenExecutor::global_attn_trace_sample_now() const {
    if (!global_attn_trace_enabled()) return false;
    const uint32_t interval =
        std::max<uint32_t>(1, env_uint32_or("QW3_ATTN_TRACE_INTERVAL", 1));
    return (global_attn_trace_seen_tokens_ % interval) == 0;
}

void QwenExecutor::global_trace_attention_layer(uint32_t layer_index,
                                                const DeviceTensor &k_cache,
                                                const DeviceTensor &q,
                                                uint32_t q_stride,
                                                const DeviceTensor &page_indices,
                                                uint32_t n_pages,
                                                uint32_t seq_len,
                                                float scale) {
    if (!global_attn_trace_enabled() || seq_len == 0) return;

    const uint32_t block_tokens =
        std::max<uint32_t>(1, env_uint32_or("QW3_ATTN_TRACE_BLOCK_TOKENS", 128));
    const uint32_t n_blocks = (seq_len + block_tokens - 1) / block_tokens;
    const uint32_t buckets = n_blocks + 1;  // final bucket should stay 0 here
    if (n_blocks == 0) return;

    if (n_blocks > global_attn_trace_block_capacity_ ||
        block_tokens != global_attn_trace_block_tokens_) {
        global_attn_trace_block_tokens_ = block_tokens;
        global_attn_trace_block_capacity_ = n_blocks;
        global_attn_trace_base_dev_ =
            backend_.tensor_i32(global_attn_trace_block_capacity_,
                                "global_attn_trace_base");
        global_attn_trace_tokens_dev_ =
            backend_.tensor_i32(global_attn_trace_block_capacity_,
                                "global_attn_trace_tokens");
        global_attn_trace_mass_ =
            backend_.tensor_f32(global_attn_trace_block_capacity_ + 1,
                                "global_attn_trace_mass");
    }
    global_attn_trace_base_host_.resize(n_blocks);
    global_attn_trace_tokens_host_.resize(n_blocks);
    for (uint32_t i = 0; i < n_blocks; ++i) {
        const uint32_t base = i * block_tokens;
        global_attn_trace_base_host_[i] = static_cast<int32_t>(base);
        const uint32_t remain = (seq_len > base) ? (seq_len - base) : 0;
        global_attn_trace_tokens_host_[i] =
            static_cast<int32_t>(std::min(block_tokens, remain));
    }
    require_status(backend_.copy_i32_from_host(
        *global_attn_trace_base_dev_, 0, global_attn_trace_base_host_.data(),
        n_blocks));
    require_status(backend_.copy_i32_from_host(
        *global_attn_trace_tokens_dev_, 0, global_attn_trace_tokens_host_.data(),
        n_blocks));

    const QwenConfig &cfg = model_.config();
    const uint32_t per_pos = cfg.n_kv_heads * cfg.head_dim;
    auto st = backend_.block_attention_mass_paged_device(
        *global_attn_trace_mass_, q, q_stride, k_cache,
        n_blocks, cfg.n_heads, cfg.n_kv_heads, per_pos, cfg.head_dim,
        *global_attn_trace_base_dev_, *global_attn_trace_tokens_dev_,
        page_indices, n_pages, kv_pages_.page_size, seq_len, scale);
    if (!st.ok) return;

    std::vector<float> mass(buckets, 0.0f);
    if (auto copy = backend_.copy_to_host(*global_attn_trace_mass_, mass.data(), 0,
                                          buckets);
        !copy.ok) {
        return;
    }

    if (!global_attn_trace_file_) {
        const char *path = std::getenv("QW3_ATTN_TRACE");
        global_attn_trace_file_ = std::fopen(path, "a");
        if (!global_attn_trace_file_) return;
    }

    double sum = 0.0;
    for (float v : mass) sum += static_cast<double>(v);

    std::FILE *f = global_attn_trace_file_;
    std::fprintf(f,
                 "{\"kind\":\"attention_mass\",\"mode\":\"global\","
                 "\"sample\":%llu,\"position\":%u,\"seq_len\":%u,"
                 "\"layer\":%u,\"n_heads\":%u,\"n_kv_heads\":%u,"
                 "\"head_dim\":%u,\"block_tokens\":%u,\"sum\":%.9g,"
                 "\"block_ids\":[",
                 static_cast<unsigned long long>(global_attn_trace_sample_),
                 position_, seq_len, layer_index, cfg.n_heads, cfg.n_kv_heads,
                 cfg.head_dim, block_tokens, sum);
    for (uint32_t i = 0; i < n_blocks; ++i) {
        if (i) std::fputc(',', f);
        std::fprintf(f, "%u", i);
    }
    if (n_blocks > 0) std::fputc(',', f);
    std::fputs("-1],\"mass\":[", f);
    for (uint32_t i = 0; i < buckets; ++i) {
        if (i) std::fputc(',', f);
        std::fprintf(f, "%.9g", static_cast<double>(mass[i]));
    }
    std::fputs("]}\n", f);
    std::fflush(f);
}

// ---- Global content-frame KV retrieval (#48/#49) ------------------------

// Build the position-invariant content-frame mean-Key index over ALL current
// (prefill) blocks, ONCE, from the pristine post-prefill cache. This is the only
// moment every block's K sits at its true baked position (orig_pos_start+tok);
// after the first assembly, selected blocks are re-RoPE'd into window slots, so
// de-RoPE'ing by their original position would no longer match the stored phase.
// The content mean is position-invariant, so building it once and keeping it is
// correct for the whole session. Reads through the FULL repository page table.
    // q8 caches cannot be de-RoPE'd and therefore leave this index unbuilt.
    // fp8 uses the typed de-RoPE/mean kernel and remains a full retrieval path.
void QwenExecutor::kvmem_build_content_index() {
    g_content_ready_ = false;
    if (!block_store_) return;
    // Internal diagnostic (default off): force the legacy window-local recency
    // path by never building the global index. Lets validation A/B the two
    // selection policies at identical fp16 cache quality.
    if (env_flag_enabled("QW3_KVMEM_RETRIEVAL_DISABLE")) return;
    const uint32_t n_blocks = block_store_->block_count();
    if (n_blocks == 0) return;

    const QwenConfig &cfg = model_.config();
    if (bs_score_layer_ < 0) {
        for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
            if (cfg.is_standard_attention_layer(il)) {
                bs_score_layer_ = static_cast<int32_t>(il);
                break;
            }
        }
    }
    if (bs_score_layer_ < 0) return;

    const uint32_t n_kv_heads = cfg.n_kv_heads;
    const uint32_t head_dim = cfg.head_dim;
    const uint32_t per_pos = n_kv_heads * head_dim;

    // block_id -> (true first position, token count) from the block store.
    const auto &blocks = block_store_->blocks();
    g_orig_base_host_.resize(n_blocks);
    g_blk_tokens_host_.resize(n_blocks);
    for (uint32_t i = 0; i < n_blocks; ++i) {
        g_orig_base_host_[i] = static_cast<int32_t>(blocks[i].orig_pos_start);
        g_blk_tokens_host_[i] = static_cast<int32_t>(blocks[i].n_tokens);
        trace_rope_position_if_out_of_range(
            "kvmem_build_content_index.kmean", blocks[i].orig_pos_start,
            blocks[i].n_tokens, cfg.n_ctx_train, bs_score_layer_);
    }

    if (n_blocks > g_kbar_global_capacity_) {
        g_kbar_global_capacity_ = n_blocks;
        g_kbar_ = kvmem_alloc_mean_index_tensor(
            static_cast<uint64_t>(g_kbar_global_capacity_) * n_kv_heads * head_dim,
            "g_kbar");
        g_score_dev_ = backend_.tensor_f32(g_kbar_global_capacity_, "g_score");
        g_orig_base_dev_ = backend_.tensor_i32(g_kbar_global_capacity_, "g_orig_base");
        g_blk_tokens_dev_ = backend_.tensor_i32(g_kbar_global_capacity_, "g_blk_tokens");
    }
    if (!g_query_content_) {
        g_query_content_ = backend_.tensor_f32(
            static_cast<uint64_t>(cfg.n_heads) * head_dim, "g_query_content");
    }
    require_status(backend_.copy_i32_from_host(
        *g_orig_base_dev_, 0, g_orig_base_host_.data(), n_blocks));
    require_status(backend_.copy_i32_from_host(
        *g_blk_tokens_dev_, 0, g_blk_tokens_host_.data(), n_blocks));

    auto st = backend_.block_kmean_content_paged_device(
        k_cache(static_cast<uint32_t>(bs_score_layer_)), *g_kbar_,
        n_blocks, n_kv_heads, per_pos, head_dim, cfg.rope_dim,
        *g_orig_base_dev_, *g_blk_tokens_dev_, kv_page_indices_device(),
        kv_pages_.page_size, cfg.rope_theta);
    if (st.ok) {
        g_content_ready_ = true;
        g_indexed_blocks_ = n_blocks;
    }

    // Per-normal-layer content index (#86): when a query-conditioned span is
    // active, build a content mean-key for EVERY normal-attention layer into
    // g_kbar_multi_ [L, n_blocks, n_kv_heads, head_dim], one reuse of the existing
    // kmean kernel per layer (slot s reads k_cache(std_layers_[s]) and writes the
    // slot-s slice). All one-time post-prefill. Feeds the mean-k softmax scorer.
    g_kbar_multi_ready_ = false;
    if (g_content_ready_ && kvmem_query_end_ > kvmem_query_begin_) {
        if (std_layers_.empty()) kvmem_resolve_std_layers();
        const uint32_t L = kvmem_qc_num_layers_;
        if (L > 0) {
            const uint64_t per_layer =
                static_cast<uint64_t>(n_blocks) * n_kv_heads * head_dim;
            if (!g_kbar_multi_ || n_blocks > g_kbar_multi_capacity_ ||
                g_kbar_multi_->count < per_layer * L) {
                g_kbar_multi_capacity_ = n_blocks;
                g_kbar_multi_ =
                    kvmem_alloc_mean_index_tensor(per_layer * L,
                                                  "g_kbar_multi");
            }
            bool all_ok = true;
            for (uint32_t s = 0; s < L && all_ok; ++s) {
                for (const KvMemBlock &block : blocks) {
                    trace_rope_position_if_out_of_range(
                        "kvmem_build_content_index.kmean_multi",
                        block.orig_pos_start, block.n_tokens, cfg.n_ctx_train,
                        static_cast<int32_t>(std_layers_[s]));
                }
                auto ml = backend_.block_kmean_content_paged_device(
                    k_cache(std_layers_[s]), *g_kbar_multi_,
                    n_blocks, n_kv_heads, per_pos, head_dim, cfg.rope_dim,
                    *g_orig_base_dev_, *g_blk_tokens_dev_, kv_page_indices_device(),
                    kv_pages_.page_size, cfg.rope_theta,
                    /*out_elem_off=*/static_cast<uint64_t>(s) * per_layer);
                if (!ml.ok) all_ok = false;
            }
            g_kbar_multi_ready_ = all_ok;
            g_kbar_multi_blocks_ = all_ok ? n_blocks : 0;
        }
    }
}

// Per decode step at the representative layer: de-RoPE the current RoPE-baked Q
// (baked at window_query_pos_) into the content frame so it can be scored against
// the content-frame mean keys at the retrieval boundary. Cheap (one launch per
// step at one layer); inert unless the global index is live.
void QwenExecutor::kvmem_snapshot_content_query(uint32_t layer_index) {
    if (!g_content_ready_) return;
    if (static_cast<int32_t>(layer_index) != bs_score_layer_) return;
    const QwenConfig &cfg = model_.config();
    trace_rope_position_if_out_of_range(
        "kvmem_snapshot_content_query.derope", window_query_pos_, 1,
        cfg.n_ctx_train, static_cast<int32_t>(layer_index));
    auto st = backend_.derope_query_device(
        *g_query_content_, *q_, /*q_stride=*/2 * cfg.head_dim,
        cfg.n_heads, cfg.head_dim, cfg.rope_dim,
        static_cast<int32_t>(window_query_pos_), cfg.rope_theta);
    g_query_ready_ = st.ok;
}

// At the retrieval boundary, score EVERY indexed block by the de-RoPE'd query vs
// its content mean key (zeroed accumulator => the score kernel OVERWRITES, since
// retrieval re-ranks fresh each interval), drain to host, and overwrite the block
// store's per-block score. pick_topk then ranks globally and can resurrect a
// block dropped from the window. Returns false if anything is not live (caller
// then keeps the window-local heat signal).
bool QwenExecutor::kvmem_retrieval_score(std::string *failure_reason) {
    if (failure_reason) failure_reason->clear();
    if (!block_store_) {
        return scorer_unavailable(failure_reason, "block_store_unavailable");
    }
    if (!g_content_ready_) {
        return scorer_unavailable(failure_reason,
                                  "single_content_index_not_ready");
    }
    if (!g_query_ready_) {
        return scorer_unavailable(failure_reason,
                                  "single_query_not_ready");
    }
    if (g_indexed_blocks_ == 0) {
        return scorer_unavailable(failure_reason,
                                  "single_index_has_zero_blocks");
    }
    const QwenConfig &cfg = model_.config();
    // Content-frame Q . k̄ has no 1/sqrt(d) softmax scale to honor (it is a bare
    // similarity rank), so use scale = 1.
    require_status(backend_.zero_tensor(*g_score_dev_));
    if (auto st = backend_.block_attn_score_step_device(
            *g_score_dev_, *g_query_content_, *g_kbar_,
            /*q_stride=*/cfg.head_dim, g_indexed_blocks_,
            cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, /*scale=*/1.0f);
        !st.ok) {
        return scorer_backend_unavailable(
            failure_reason, "single_scorer_kernel_failed", st);
    }
    std::vector<float> score(g_indexed_blocks_, 0.0f);
    if (auto st = backend_.copy_to_host(*g_score_dev_, score.data(), 0,
                                        g_indexed_blocks_);
        !st.ok) {
        return scorer_backend_unavailable(
            failure_reason, "single_score_d2h_failed", st);
    }
    // Bare content-similarity ranks (Q·k̄); softmax is monotonic so top-k selection
    // is identical with or without it — feed the raw scores straight through.
    std::vector<double> scores(block_store_->block_count(), 0.0);
    for (uint32_t id = 0; id < g_indexed_blocks_ && id < scores.size(); ++id) {
        scores[id] = static_cast<double>(score[id]);
    }
    block_store_->set_retrieval_scores(scores);
    if (std::getenv("QW3_KVMEM_TRACE")) {
        std::vector<std::pair<float, uint32_t>> ranked;
        ranked.reserve(g_indexed_blocks_);
        for (uint32_t id = 0; id < g_indexed_blocks_; ++id) {
            ranked.emplace_back(score[id], id);
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto &a, const auto &b) { return a.first > b.first; });
        std::fprintf(stderr, "[bs-retrieval] top blocks: ");
        for (size_t i = 0; i < ranked.size() && i < 6; ++i) {
            std::fprintf(stderr, "blk%u=%.3f ", ranked[i].second, ranked[i].first);
        }
        std::fprintf(stderr, "(indexed=%u)\n", g_indexed_blocks_);
    }
    return true;
}

void QwenExecutor::kvmem_set_trace_metadata(
    const std::string &trace_tag, uint32_t context_begin, uint32_t context_end,
    const std::vector<uint32_t> &prompt_tokens) {
    kvmem_trace_tag_ = trace_tag;
    kvmem_context_begin_ = context_begin;
    kvmem_context_end_ = context_end;
    // Avoid copying a potentially 250K-token prompt in ordinary inference or
    // legacy score dumps. Tagged context-overlap diagnostics opt in explicitly.
    const bool valid_context_span = context_end > context_begin &&
        context_end <= prompt_tokens.size();
    const bool whole_prompt_trace = context_begin == 0 && context_end == 0;
    if (std::getenv("QW3_KVMEM_DUMP_SCORES") && !trace_tag.empty() &&
        (valid_context_span || whole_prompt_trace)) {
        kvmem_trace_prompt_tokens_ = prompt_tokens;
    } else {
        kvmem_trace_prompt_tokens_.clear();
    }
}

// Mark the final user message's token span [begin,end) as the question. Called
// by the backend BEFORE prefill when query-conditioned KVMem is enabled. Resets
// the capture state and ensures bs_score_layer_ + the [M, n_heads, head_dim]
// content-frame buffer are ready so kvmem_capture_query_multi can de-RoPE the
// in-span Q rows during prefill. begin==end (default) leaves everything inert
// -> the single-token retrieval / recency path runs unchanged.
void QwenExecutor::kvmem_set_query_span(uint32_t begin, uint32_t end,
                                        uint32_t prompt_tokens,
                                        bool preserve_content_index,
                                        bool capture_content_without_query) {
    // A previous request may have left the final asynchronous capture copy in
    // flight. Drain it before changing the span or reusing either bounce slot.
    kvmem_drain_query_capture();
    kvmem_query_host_capture_ = false;
    // Consume + clear the resume base up front (set by the preceding
    // kvmem_truncate_to on a session-continuation turn). It is only USED when a
    // span is active below (above-budget QC); on a cold/below-budget call it is
    // still cleared here so it can never leak into a later turn.
    const uint32_t resume_base = kvmem_qc_resume_base_tokens_;
    kvmem_qc_resume_base_tokens_ = 0;
    const uint32_t preserved_captured_blocks = kvmem_qc_captured_blocks_;
    const uint32_t preserved_captured_tokens = kvmem_qc_captured_tokens_;
    const uint32_t preserved_stride_blocks = kvmem_qc_layer_stride_blocks_;
    const uint32_t preserved_subblocks = kvmem_qc_n_subblocks_;
    kvmem_query_begin_ = begin;
    kvmem_query_end_ = end;
    g_query_multi_count_ = 0;
    g_query_multi_ready_ = false;
    kvmem_qc_capture_active_ =
        end > begin || capture_content_without_query;
    g_kbar_multi_ready_ = false;
    g_kbar_multi_blocks_ = 0;
    if (!preserve_content_index) {
        kvmem_qc_total_blocks_ = 0;
        kvmem_qc_prompt_tokens_ = 0;
        kvmem_qc_captured_blocks_ = 0;
        kvmem_qc_captured_tokens_ = 0;
    } else {
        // The previously published prefix is no longer the complete scoreable
        // range once teacher-forced transcript tokens resume. Keep its immutable
        // per-layer slices, but unpublish until the next query boundary.
        g_content_ready_ = false;
        g_indexed_blocks_ = 0;
    }
    g_kraw_multi_ready_ = false;
    kvmem_qc_total_tokens_ = 0;
    if (!kvmem_qc_capture_active_) return;
    const QwenConfig &cfg = model_.config();
    // Resolve the normal-attention layer set (pins bs_score_layer_ as a side
    // effect) and read the env A/B knobs so capture/scoring agree on L this run.
    kvmem_resolve_std_layers();
    // g_query_multi_ holds the per-layer de-RoPE'd question rows, laid out
    // [L, S, n_heads, head_dim] (S = span length, per-layer row stride). Allocate
    // by the exact L*S so a tiny question only costs tens of MB.
    const uint32_t S = std::max<uint32_t>(
        1, kvmem_query_end_ - kvmem_query_begin_);
    kvmem_query_span_ = S;
    const uint32_t L = std::max<uint32_t>(kvmem_qc_num_layers_, 1u);
    const uint64_t rows = static_cast<uint64_t>(L) * S;
    const uint64_t query_row_elems =
        static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim;
    if (rows > UINT64_MAX / query_row_elems) {
        throw std::runtime_error("KVMem query capture size overflow");
    }
    // Default to FP16 query capture for CUDA mean-K. Queries that fit in the
    // bounded scoring stage stay entirely on GPU; only longer queries use the
    // host-backed streaming path. ExactMass and the archived DeltaNet scorer
    // need their original full-GPU FP32 layouts; clean-query PASS-A/PASS-B also
    // retains its legacy FP32 stash semantics.
    const uint32_t score_tokens = std::min<uint32_t>(
        S, env_uint32_or("QW3_KVMEM_QUERY_SCORE_CHUNK", 256));
    const bool fp16_mean_query =
        env_flag_enabled("QW3_KVMEM_QUERY_HOST_CAPTURE", true) &&
        std::strcmp(backend_.name(), "cuda-device") == 0 &&
        kvmem_query_end_ > kvmem_query_begin_ &&
        !kvmem_qc_pertoken_ && !kvmem_qc_deltanet_ &&
        !env_flag_enabled("QW3_KVMEM_CLEAN_QUERY");
    kvmem_query_host_capture_ = fp16_mean_query && S > score_tokens;
    if (kvmem_query_host_capture_) {
        const uint64_t elems = rows * query_row_elems;
        // Pageable storage avoids consuming the process' finite pinned-memory
        // budget for an arbitrarily long user query. Two small pinned bounce
        // buffers below provide asynchronous D2H/H2D.
        const bool shrink_host =
            g_query_multi_host_ && elems > 0 &&
            g_query_multi_host_capacity_ / 4 > elems;
        if (!g_query_multi_host_ ||
            g_query_multi_host_capacity_ < elems || shrink_host) {
            g_query_multi_host_.reset(new uint16_t[elems]);
            g_query_multi_host_capacity_ = elems;
        }
        const uint64_t score_rows =
            static_cast<uint64_t>(L) * std::max<uint32_t>(score_tokens, 1u);
        // Drop a former full-query allocation rather than letting CUDA retain
        // its high-water mark after switching to bounded streaming.
        if (!g_query_multi_ ||
            g_query_multi_->elem_size != sizeof(uint16_t) ||
            g_query_multi_capacity_ < score_rows ||
            g_query_multi_capacity_ > score_rows * 2) {
            g_query_multi_ = backend_.tensor_f16(
                score_rows * query_row_elems, "g_query_score_stage");
            g_query_multi_capacity_ = score_rows;
        }
        if (std::getenv("QW3_KVMEM_TRACE")) {
            std::fprintf(
                stderr,
                "[bs-qcap-host] enabled=1 layers=%u tokens=%u "
                "host_mib=%.2f score_chunk=%u gpu_stage_mib=%.2f\n",
                L, S,
                static_cast<double>(elems * sizeof(uint16_t)) /
                    (1024.0 * 1024.0),
                score_tokens,
                static_cast<double>(
                    score_rows * query_row_elems * sizeof(uint16_t)) /
                    (1024.0 * 1024.0));
        }
    } else {
        const uint32_t query_elem_size =
            fp16_mean_query ? sizeof(uint16_t) : sizeof(float);
        if (!g_query_multi_ || g_query_multi_capacity_ < rows ||
            g_query_multi_->elem_size != query_elem_size) {
            g_query_multi_ = fp16_mean_query
                ? backend_.tensor_f16(
                      rows * query_row_elems, "g_query_multi_fp16")
                : backend_.tensor_f32(
                      rows * query_row_elems, "g_query_multi");
            g_query_multi_capacity_ = rows;
        }
        if (fp16_mean_query && std::getenv("QW3_KVMEM_TRACE")) {
            std::fprintf(
                stderr,
                "[bs-qcap-gpu] fp16=1 layers=%u tokens=%u "
                "gpu_mib=%.2f threshold=%u\n",
                L, S,
                static_cast<double>(
                    rows * query_row_elems * sizeof(uint16_t)) /
                    (1024.0 * 1024.0),
                score_tokens);
        }
    }
    // Incremental full-coverage content index (#91): the paged builder can only
    // run once from the pristine cache and misses the tail of histories larger
    // than the GPU page pool. Instead capture each block's per-layer content
    // mean-key the moment its K is RoPE'd during prefill, covering EVERY block.
    // Fix the per-layer stride (final block count) up front from the prompt length
    // so each (layer, chunk) slice lands at a stable offset. Pre-size all buffers
    // for the full block count so capture + scoring never reallocate mid-prefill.
    if (!block_store_ || prompt_tokens == 0) return;
    const uint32_t bt = std::max<uint32_t>(block_store_->config().block_tokens, 1u);
    const uint32_t total_blocks = (prompt_tokens + bt - 1) / bt;
    if (total_blocks == 0) return;
    kvmem_qc_total_blocks_ = total_blocks;
    kvmem_qc_prompt_tokens_ = prompt_tokens;
    // Sub-block mean-k (SubBlockMeanK): capture this many equal, non-overlapping
    // sub-block means per block so the scorer can run softmax at sub-block
    // granularity. 1 for plain mean-k / per-token -> byte-identical layout.
    kvmem_qc_n_subblocks_ =
        std::max<uint32_t>(1u, block_store_->config().n_subblocks);
    kvmem_qc_subblock_max_ =
        block_store_->config().subblock_reduce == KvMemSubblockReduce::Max;
    if (kvmem_qc_n_subblocks_ > 1 && std::getenv("QW3_KVMEM_TRACE")) {
        std::fprintf(stderr,
                     "[bs-subblock] sub-block mean-k active: n_subblocks=%u "
                     "(sub_tokens=%u per %u-token block) reduce=%s\n",
                     kvmem_qc_n_subblocks_,
                     (bt + kvmem_qc_n_subblocks_ - 1) / kvmem_qc_n_subblocks_, bt,
                     kvmem_qc_subblock_max_ ? "max" : "sum");
    }
    const uint32_t n_kv_heads = cfg.n_kv_heads;
    const uint32_t head_dim = cfg.head_dim;
    // Fixed per-layer stride for g_kbar_multi_: pin it at the session's ctx block
    // capacity (ceil(kv_ctx_size_/block_tokens)) so preserved [0,D) index slices
    // stay at a stable per-layer offset as the block count grows across resumed
    // turns (server-side session continuation). This decouples "where layer l's
    // slice starts" from "how many blocks this turn scores" (total_blocks). The
    // scorer + incremental capture both key on this stride; the buffer is
    // over-allocated to the ctx cap once (≈256 MiB at 262144 ctx with the
    // 16-layer FP16 mean index used by Qwen3.6-27B) and never
    // reallocates as the session grows. Clamp up so a prompt longer than the
    // configured ctx can't under-size the stride.
    const uint32_t ctx_blocks = (kv_ctx_size_ + bt - 1) / bt;
    const uint32_t stride_blocks = std::max(ctx_blocks, total_blocks);
    if (preserve_content_index &&
        (kvmem_qc_pertoken_ || kvmem_qc_deltanet_ ||
         !g_kbar_multi_ || preserved_stride_blocks != stride_blocks ||
         preserved_subblocks != kvmem_qc_n_subblocks_ ||
         stride_blocks > g_kbar_multi_capacity_)) {
        throw std::runtime_error(
            "KVMem transcript replay cannot preserve the existing mean-K "
            "content index");
    }
    kvmem_qc_layer_stride_blocks_ = stride_blocks;
    const uint64_t per_layer = static_cast<uint64_t>(stride_blocks) *
                               kvmem_qc_n_subblocks_ * n_kv_heads * head_dim;
    // Per-layer content index [L, stride_blocks, n_subblocks, n_kv_heads,
    // head_dim]. Production storage is always FP16 even when the active KV
    // cache is FP8; builders and scorers still accumulate in FP32. FP32 KV is
    // retained as a diagnostic full-precision index mode.
    bool freshly_allocated = false;
    if (!g_kbar_multi_ || stride_blocks > g_kbar_multi_capacity_ ||
        g_kbar_multi_->count < per_layer * L) {
        g_kbar_multi_capacity_ = stride_blocks;
        g_kbar_multi_ =
            kvmem_alloc_mean_index_tensor(per_layer * L, "g_kbar_multi");
        freshly_allocated = true;
        if (std::getenv("QW3_KVMEM_TRACE")) {
            const char *index_dtype =
                g_kbar_multi_->elem_size == sizeof(float) ? "fp32" : "fp16";
            const uint64_t elements = per_layer * L;
            const uint64_t bytes =
                elements * static_cast<uint64_t>(g_kbar_multi_->elem_size);
            std::fprintf(
                stderr,
                "[bs-kbar-index] dtype=%s elem_bytes=%u layers=%u "
                "stride_blocks=%u subblocks=%u elements=%llu bytes=%llu "
                "gib=%.3f\n",
                index_dtype, g_kbar_multi_->elem_size, L, stride_blocks,
                kvmem_qc_n_subblocks_,
                static_cast<unsigned long long>(elements),
                static_cast<unsigned long long>(bytes),
                static_cast<double>(bytes) /
                    (1024.0 * 1024.0 * 1024.0));
        }
    }
    // Resume seeding keeps the exact token prefix represented by the existing
    // index. A suffix beginning mid-block now merges its new K rows with that
    // block's prior partial mean instead of skipping the boundary block.
    if (preserve_content_index) {
        if (freshly_allocated) {
            throw std::runtime_error(
                "KVMem transcript replay unexpectedly reallocated its content index");
        }
        kvmem_qc_captured_blocks_ =
            std::min(preserved_captured_blocks, total_blocks);
        kvmem_qc_captured_tokens_ =
            std::min(preserved_captured_tokens, prompt_tokens);
    } else if (resume_base > 0 && resume_base <= prompt_tokens &&
               !freshly_allocated) {
        kvmem_qc_captured_tokens_ = resume_base;
        kvmem_qc_captured_blocks_ = (resume_base + bt - 1) / bt;
    } else {
        kvmem_qc_captured_tokens_ = 0;
        require_status(backend_.zero_tensor(*g_kbar_multi_));
    }
    // Raw-key ExactMass (per-token method): keep the full per-token de-RoPE'd K
    // (no mean) so the scorer can softmax over a block's tokens. Layout
    // [L, total_tokens, n_kv_heads, head_dim] fp32 — at max history (~113K tok × 16
    // layers) this is ~7.4 GB, so it is allocated ONLY under --kvmem-retrieval-method
    // per-token (the default mean-k path pays zero).
    if (kvmem_qc_pertoken_) {
        kvmem_qc_total_tokens_ = prompt_tokens;
        const uint64_t kraw_per_layer =
            static_cast<uint64_t>(prompt_tokens) * n_kv_heads * head_dim;
        const uint64_t kraw_total = kraw_per_layer * L;
        if (!g_kraw_multi_ || g_kraw_multi_->count < kraw_total) {
            g_kraw_multi_ = backend_.tensor_f32(kraw_total, "g_kraw_multi");
        }
        if (std::getenv("QW3_KVMEM_TRACE")) {
            std::fprintf(stderr,
                "[bs-exactmass] alloc g_kraw_multi: L=%u total_tokens=%u -> %.2f GiB\n",
                L, prompt_tokens,
                static_cast<double>(kraw_total) * sizeof(float) / (1024.0*1024.0*1024.0));
        }
    }
    // Single-layer index buffer (g_kbar_) + the per-block score buffer
    // (g_score_dev_, sized by g_kbar_global_capacity_) + the content query slot
    // (g_query_content_) back the single-layer multitoken fallback and the
    // slot-0 copy at capture completion. Size them for the full block count.
    if (total_blocks > g_kbar_global_capacity_) {
        g_kbar_global_capacity_ = total_blocks;
        g_kbar_ = kvmem_alloc_mean_index_tensor(per_layer, "g_kbar");
        g_score_dev_ = backend_.tensor_f32(g_kbar_global_capacity_, "g_score");
        g_orig_base_dev_ = backend_.tensor_i32(g_kbar_global_capacity_, "g_orig_base");
        g_blk_tokens_dev_ = backend_.tensor_i32(g_kbar_global_capacity_, "g_blk_tokens");
    }
    if (!g_query_content_) {
        g_query_content_ = backend_.tensor_f32(
            static_cast<uint64_t>(cfg.n_heads) * head_dim, "g_query_content");
    }

    // DeltaNet-state retrieval buffers (deltanet_retrieval.md). Allocated only
    // when the deltanet method is selected. Sized for the full block count.
    if (kvmem_qc_deltanet_) {
        kvmem_resolve_deltanet_layers();
        kvmem_dn_ready_ = false;
        kvmem_dn_qcount_ = 0;
        const uint32_t L_dn = kvmem_dn_num_layers_;
        if (L_dn > 0) {
            const uint32_t vh = cfg.num_v_heads();
            const uint32_t dv = cfg.head_v_dim_ssm();
            const uint32_t dk = cfg.head_k_dim();
            // g_deltanet_snap_ [L_dn, blocks, vh, dv, dk] fp32 (S_j per block).
            const uint64_t snap_per_layer =
                static_cast<uint64_t>(total_blocks) * vh * dv * dk;
            // g_deltanet_decaysum_ / decay_d / r [L_dn, blocks, vh] fp32.
            const uint64_t small_per_layer =
                static_cast<uint64_t>(total_blocks) * vh;
            if (!g_deltanet_snap_ || total_blocks > g_deltanet_capacity_blocks_) {
                g_deltanet_capacity_blocks_ = total_blocks;
                g_deltanet_snap_ =
                    backend_.tensor_f32(snap_per_layer * L_dn, "g_deltanet_snap");
                g_deltanet_decaysum_ =
                    backend_.tensor_f32(small_per_layer * L_dn, "g_deltanet_decaysum");
                g_deltanet_decay_d_ =
                    backend_.tensor_f32(small_per_layer * L_dn, "g_deltanet_decay_d");
                g_deltanet_r_ =
                    backend_.tensor_f32(small_per_layer * L_dn, "g_deltanet_r");
            }
            // g_deltanet_q_ [L_dn, S, num_k_heads, dk] fp32.
            const uint64_t q_rows = static_cast<uint64_t>(L_dn) * S;
            const uint64_t q_elems = q_rows * cfg.num_k_heads() * dk;
            if (!g_deltanet_q_ || g_deltanet_q_->count < q_elems) {
                g_deltanet_q_ = backend_.tensor_f32(q_elems, "g_deltanet_q");
            }
            // Zero the snapshot + decay accumulators so uncaptured blocks read 0.
            require_status(backend_.zero_tensor(*g_deltanet_snap_));
            require_status(backend_.zero_tensor(*g_deltanet_decaysum_));
        }
    }
}

bool QwenExecutor::kvmem_publish_captured_prefix(uint32_t scoreable_tokens) {
    if (!kvmem_enabled_ || !block_store_ || !g_kbar_multi_) return false;
    if (kvmem_qc_pertoken_ || kvmem_qc_deltanet_) return false;
    const uint32_t bt = std::max<uint32_t>(
        1, block_store_->config().block_tokens);
    // Score only sealed context blocks before the replay boundary. The live
    // first-pass query suffix is pinned by kvmem_set_pin_from_block() and must
    // not dilute the softmax or publish temporary K means.
    const uint32_t n_blocks = std::min<uint32_t>(
        block_store_->block_count(), scoreable_tokens / bt);
    if (n_blocks == 0 || scoreable_tokens > kvmem_qc_captured_tokens_ ||
        n_blocks > kvmem_qc_captured_blocks_ ||
        n_blocks > kvmem_qc_total_blocks_) {
        return false;
    }
    const QwenConfig &cfg = model_.config();
    const uint64_t per_pos =
        static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
    const uint64_t slot0_elems = static_cast<uint64_t>(n_blocks) * per_pos;
    if (kvmem_qc_n_subblocks_ == 1 && g_kbar_ &&
        g_kbar_->count >= slot0_elems) {
        require_status(backend_.copy_d2d(*g_kbar_, *g_kbar_multi_, 0,
                                         slot0_elems));
    }
    g_content_ready_ = true;
    g_indexed_blocks_ = n_blocks;
    g_kbar_multi_ready_ = true;
    g_kbar_multi_blocks_ = n_blocks;
    if (std::getenv("QW3_KVMEM_TRACE")) {
        std::fprintf(stderr,
                     "[bs-kbar-build] publish transcript prefix: blocks=%u "
                     "captured=%u captured_tokens=%u scoreable_tokens=%u "
                     "total=%u query_ready=%d\n",
                     n_blocks, kvmem_qc_captured_blocks_,
                     kvmem_qc_captured_tokens_, scoreable_tokens,
                     kvmem_qc_total_blocks_, g_query_multi_ready_ ? 1 : 0);
    }
    return g_query_multi_ready_;
}

// Resolve the normal-attention layer slot maps from the model config + env A/B
// knobs (once per query span). std_layers_[slot] -> layer id; std_layer_slot_[il]
// -> slot 0..L-1 (or -1 for DeltaNet / linear layers). Default: all normal
// layers (full_attention_interval over n_layers). QW3_KVMEM_QC_SINGLE_LAYER=1 ->
// L=1 (only bs_score_layer_, the legacy single-layer multitoken path).
// QW3_KVMEM_QC_LAYERS=N caps L to the first N normal layers (debug).
void QwenExecutor::kvmem_resolve_std_layers() {
    const QwenConfig &cfg = model_.config();
    if (bs_score_layer_ < 0) {
        for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
            if (cfg.is_standard_attention_layer(il)) {
                bs_score_layer_ = static_cast<int32_t>(il);
                break;
            }
        }
    }
    kvmem_no_rerope_ = env_flag_enabled("QW3_KVMEM_NO_REROPE");
    // Default ON: record the ACTUAL window bake position as baked_pos for chunks
    // prefilled while the window is active, so downstream de-rotate (set_selection
    // -> assemble, canonicalize) cancels the real rotation. QW3_KVMEM_FIX_BAKEDPOS=0
    // restores the old (buggy true-pos) bookkeeping for A/B comparison.
    if (const char *e = std::getenv("QW3_KVMEM_FIX_BAKEDPOS")) {
        kvmem_fix_bakedpos_ = !(e[0] == '0' && e[1] == '\0');
    }
    // Query-conditioned scorer choice comes from the CLI (--kvmem-retrieval-method):
    // PerToken -> raw-key ExactMass (needs the ~7.4 GB g_kraw_multi_ buffer), else
    // MeanK -> softmax-over-pages on the ~28 MB mean-key buffer (default).
    kvmem_qc_pertoken_ = block_store_ && block_store_->config().retrieval_method ==
                                             KvMemRetrievalMethod::PerToken;
    kvmem_qc_deltanet_ = block_store_ && block_store_->config().retrieval_method ==
                                             KvMemRetrievalMethod::DeltaNet;
    kvmem_qc_layer_cap_ = -1;
    if (const char *cap = std::getenv("QW3_KVMEM_QC_LAYERS")) {
        const int v = std::atoi(cap);
        if (v > 0) kvmem_qc_layer_cap_ = v;
    }
    std_layers_.clear();
    std_layer_slot_.assign(weights_.n_layers(), -1);
    for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
        if (!cfg.is_standard_attention_layer(il)) continue;
        if (kvmem_qc_layer_cap_ >= 0 &&
            static_cast<int32_t>(std_layers_.size()) >= kvmem_qc_layer_cap_) {
            break;
        }
        std_layer_slot_[il] = static_cast<int32_t>(std_layers_.size());
        std_layers_.push_back(il);
    }
    kvmem_qc_num_layers_ = static_cast<uint32_t>(std_layers_.size());
}

// Resolve the DeltaNet (recurrent) layer subset that feeds DeltaNet-state
// retrieval (deltanet_retrieval.md). The per-block state edit E_j is a d_v*d_k
// fp32 matrix per (layer, head), so the number of DeltaNet layers is capped by a
// memory budget: pick evenly-spaced recurrent layers up to the configured count
// (deltanet_layers, 0 => half the DeltaNet layers), then clamp so the E_j
// snapshot buffer fits deltanet_mem_budget_bytes. Logs the resolved L_dn + the
// estimated buffer size; hard-errors if even one layer exceeds the budget.
void QwenExecutor::kvmem_resolve_deltanet_layers() {
    const QwenConfig &cfg = model_.config();
    dn_layers_.clear();
    dn_layer_slot_.assign(weights_.n_layers(), -1);
    kvmem_dn_num_layers_ = 0;
    if (!block_store_) return;

    // Enumerate all recurrent (DeltaNet) layers.
    std::vector<uint32_t> recurrent;
    for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
        if (weights_.layer(il).recurrent) recurrent.push_back(il);
    }
    const uint32_t n_dn = static_cast<uint32_t>(recurrent.size());
    if (n_dn == 0) return;

    const KvMemStoreConfig &bc = block_store_->config();
    // Desired count: explicit flag, else half the DeltaNet layers (>=1).
    uint32_t desired = bc.deltanet_layers;
    if (desired == 0) desired = std::max<uint32_t>(1u, n_dn / 2);
    desired = std::min(desired, n_dn);

    // Budget clamp. E_j snapshot bytes per layer = blocks * v_heads * d_v * d_k * 4.
    const uint32_t bt = std::max<uint32_t>(bc.block_tokens, 1u);
    const uint32_t ctx_blocks = (kv_ctx_size_ + bt - 1) / bt;
    const uint64_t bytes_per_layer =
        static_cast<uint64_t>(ctx_blocks) * cfg.num_v_heads() *
        cfg.head_v_dim_ssm() * cfg.head_k_dim() * sizeof(float);
    uint64_t budget = bc.deltanet_mem_budget_bytes;
    if (budget == 0) budget = 32ull * 1024 * 1024 * 1024;  // 32 GiB default
    uint32_t max_by_budget = bytes_per_layer > 0
        ? static_cast<uint32_t>(budget / bytes_per_layer)
        : desired;
    if (max_by_budget == 0) {
        throw std::runtime_error(
            "kvmem deltanet retrieval: a single DeltaNet layer's state-edit buffer (" +
            std::to_string(bytes_per_layer / (1024 * 1024)) +
            " MiB) exceeds --kvmem-deltanet-mem-budget-gb; lower the context size, "
            "raise the budget, or increase --kvmem-block-tokens");
    }
    const uint32_t L_dn = std::min(desired, max_by_budget);

    // Select either evenly across depth (the original behavior) or take the
    // deepest recurrent layers. The latter is useful for retrieval experiments:
    // late queries/edits are more semantic, and a small late-only subset avoids
    // giving early noisy layers equal weight after per-layer RMS normalization.
    for (uint32_t k = 0; k < L_dn; ++k) {
        uint32_t idx = 0;
        if (bc.deltanet_layer_policy == KvMemDeltaNetLayerPolicy::Late) {
            idx = n_dn - L_dn + k;
        } else {
            idx = (L_dn == 1) ? (n_dn / 2)
                              : static_cast<uint32_t>(
                                    (static_cast<uint64_t>(k) * (n_dn - 1)) /
                                    (L_dn - 1));
        }
        if (idx >= n_dn) idx = n_dn - 1;
        const uint32_t il = recurrent[idx];
        if (dn_layer_slot_[il] >= 0) continue;   // dedupe (rounding collision)
        dn_layer_slot_[il] = static_cast<int32_t>(dn_layers_.size());
        dn_layers_.push_back(il);
    }
    kvmem_dn_num_layers_ = static_cast<uint32_t>(dn_layers_.size());

    if (std::getenv("QW3_KVMEM_TRACE") || std::getenv("QW3_KVMEM_DELTANET_TRACE")) {
        std::fprintf(stderr,
            "[bs-deltanet] resolved L_dn=%u of %u DeltaNet layers "
            "(desired=%u budget_cap=%u policy=%s), v_heads=%u state=%ux%u, "
            "~%.2f GiB E_j buffer (%u ctx blocks)\n",
            kvmem_dn_num_layers_, n_dn, desired, max_by_budget,
            bc.deltanet_layer_policy == KvMemDeltaNetLayerPolicy::Late
                ? "late" : "even",
            cfg.num_v_heads(), cfg.head_v_dim_ssm(), cfg.head_k_dim(),
            static_cast<double>(bytes_per_layer) * kvmem_dn_num_layers_ /
                (1024.0 * 1024.0 * 1024.0),
            ctx_blocks);
        std::fprintf(stderr, "[bs-deltanet] layer ids:");
        for (uint32_t il : dn_layers_) std::fprintf(stderr, " %u", il);
        std::fprintf(stderr, "\n");
    }
}

// DeltaNet-state retrieval query capture. After recurrent_batch for a selected
// DeltaNet layer, conv_out holds the L2-normalized DeltaNet Q in the first
// num_k_heads*head_k_dim of each row. Copy the in-span question rows into
// g_deltanet_q_[dn_slot, count..] (no RoPE — DeltaNet queries are not rotated).
// Mirrors kvmem_capture_query_multi's span-overlap bookkeeping; advances the
// shared row count once (after the last DeltaNet slot writes) and publishes the
// index when the whole span is captured across all layers.
void QwenExecutor::kvmem_capture_deltanet_query(uint32_t dn_slot,
                                                uint32_t chunk_off,
                                                uint32_t batch,
                                                uint32_t base_pos,
                                                const DeviceTensor &conv_out,
                                                uint32_t conv_stride) {
    if (kvmem_query_end_ <= kvmem_query_begin_) return;
    if (!g_deltanet_q_ || kvmem_dn_ready_) return;
    (void)chunk_off;
    const QuerySpanOverlap overlap = query_span_overlap(
        base_pos, batch, kvmem_query_begin_, kvmem_query_end_);
    if (overlap.count == 0) return;
    const uint32_t r0 = overlap.row_offset;    // chunk-local first in-span row
    const uint32_t cnt = overlap.count;        // rows captured this chunk
    const QwenConfig &cfg = model_.config();
    const uint32_t num_k_heads = cfg.num_k_heads();
    const uint32_t head_k_dim = cfg.head_k_dim();
    const uint32_t S = kvmem_query_span_;
    const uint64_t q_dst_off =
        (static_cast<uint64_t>(dn_slot) * S + kvmem_dn_qcount_) *
        num_k_heads * head_k_dim;
    auto st = backend_.deltanet_pack_query_device(
        *g_deltanet_q_, conv_out, q_dst_off, conv_stride, r0,
        /*dst_row_base=*/0, cnt, num_k_heads, head_k_dim);
    if (!st.ok) return;
    // Advance the shared per-slot row count once, after the last DeltaNet slot.
    const uint32_t L = std::max<uint32_t>(kvmem_dn_num_layers_, 1u);
    if (dn_slot + 1 == L) {
        kvmem_dn_qcount_ += cnt;
        if (kvmem_dn_qcount_ >= S) kvmem_dn_ready_ = true;
        if (std::getenv("QW3_KVMEM_DELTANET_TRACE")) {
            std::fprintf(stderr,
                "[bs-deltanet-qcap] slot=%u r0=%u cnt=%u count=%u/%u ready=%d\n",
                dn_slot, r0, cnt, kvmem_dn_qcount_, S, kvmem_dn_ready_ ? 1 : 0);
        }
    }
}

// DeltaNet-state retrieval boundary scorer (deltanet_retrieval.md §4-10). Folds
// the captured per-block state edits E_j = S_j - a_j S_{j-1}, in-block decays a_j,
// and DeltaNet queries into the per-block score:
//   s_j = Σ_l w_l · RMSnorm_j[ TopKMean_h( TopKMean_t( d_j·||E_j^T q_t||_2 ) ) ]
// with d_j = exp(G_M - G_j) (or 1 when deltanet_decay is off). Returns false
// (caller falls back to mean-k) if the capture isn't live.
bool QwenExecutor::kvmem_retrieval_score_deltanet(
        std::string *failure_reason) {
    if (failure_reason) failure_reason->clear();
    if (!block_store_) {
        return scorer_unavailable(failure_reason, "block_store_unavailable");
    }
    if (!kvmem_qc_deltanet_) {
        return scorer_unavailable(failure_reason,
                                  "deltanet_not_configured");
    }
    if (!kvmem_dn_ready_) {
        return scorer_unavailable(failure_reason,
                                  "deltanet_capture_not_ready");
    }
    if (kvmem_dn_num_layers_ == 0) {
        return scorer_unavailable(failure_reason,
                                  "deltanet_has_zero_layers");
    }
    if (!g_deltanet_snap_ || !g_deltanet_decaysum_ || !g_deltanet_decay_d_ ||
        !g_deltanet_q_ || !g_deltanet_r_) {
        return scorer_unavailable(failure_reason,
                                  "deltanet_buffer_unavailable");
    }
    const QwenConfig &cfg = model_.config();
    const uint32_t L = kvmem_dn_num_layers_;
    const uint32_t nb = kvmem_qc_total_blocks_;
    const uint32_t vh = cfg.num_v_heads();
    const uint32_t dv = cfg.head_v_dim_ssm();
    const uint32_t dk = cfg.head_k_dim();
    const uint32_t nkh = cfg.num_k_heads();
    const uint32_t M = std::min(kvmem_dn_qcount_, kvmem_query_span_);
    if (nb == 0) {
        return scorer_unavailable(failure_reason,
                                  "deltanet_index_has_zero_blocks");
    }
    if (vh == 0) {
        return scorer_unavailable(failure_reason,
                                  "deltanet_has_zero_value_heads");
    }
    if (M == 0) {
        return scorer_unavailable(failure_reason,
                                  "deltanet_query_capture_empty");
    }
    const KvMemStoreConfig &bc = block_store_->config();
    const uint32_t topk_q = std::max<uint32_t>(1u, bc.deltanet_topk_q);
    const uint32_t topk_h = std::max<uint32_t>(1u, bc.deltanet_topk_h);

    const uint64_t small_per_layer = static_cast<uint64_t>(nb) * vh;
    const uint64_t snap_per_layer =
        static_cast<uint64_t>(nb) * vh * dv * dk;
    const uint64_t q_per_layer =
        static_cast<uint64_t>(kvmem_query_span_) * nkh * dk;

    // Build the post-block decay coefficient d[l,blk,vh] = exp(G_M - G_j) on host
    // from the accumulated in-block log-decays (G_j = Σ_{u<=j} decaysum[u]). This
    // is a cheap [L*nb*vh] reduction; done on host for clarity (first version).
    std::vector<float> decaysum(small_per_layer * L, 0.0f);
    if (auto cp = backend_.copy_to_host(*g_deltanet_decaysum_, decaysum.data(), 0,
                                        decaysum.size());
        !cp.ok) {
        return scorer_backend_unavailable(
            failure_reason, "deltanet_decaysum_d2h_failed", cp);
    }
    std::vector<float> decay_d(small_per_layer * L, 1.0f);
    if (bc.deltanet_decay) {
        for (uint32_t l = 0; l < L; ++l) {
            for (uint32_t h = 0; h < vh; ++h) {
                // G_j prefix sum over blocks; d_j = exp(G_M - G_j).
                double G = 0.0;
                std::vector<double> Gj(nb, 0.0);
                for (uint32_t j = 0; j < nb; ++j) {
                    G += decaysum[(static_cast<uint64_t>(l) * nb + j) * vh + h];
                    Gj[j] = G;
                }
                const double G_M = Gj[nb - 1];
                for (uint32_t j = 0; j < nb; ++j) {
                    decay_d[(static_cast<uint64_t>(l) * nb + j) * vh + h] =
                        static_cast<float>(std::exp(G_M - Gj[j]));
                }
            }
        }
    }
    if (auto cp = backend_.copy_bytes_from_host(
            *g_deltanet_decay_d_, /*byte_offset=*/0, decay_d.data(),
            decay_d.size() * sizeof(float));
        !cp.ok) {
        return scorer_backend_unavailable(
            failure_reason, "deltanet_decay_h2d_failed", cp);
    }

    // Per-layer block/head scores r[l,blk,vh] via the GPU scorer.
    for (uint32_t l = 0; l < L; ++l) {
        if (auto st = backend_.deltanet_block_score_device(
                *g_deltanet_r_, *g_deltanet_snap_, *g_deltanet_decaysum_,
                *g_deltanet_decay_d_, *g_deltanet_q_,
                /*r_off=*/static_cast<uint64_t>(l) * small_per_layer,
                /*snap_off=*/static_cast<uint64_t>(l) * snap_per_layer,
                /*decay_off=*/static_cast<uint64_t>(l) * small_per_layer,
                /*decay_d_off=*/static_cast<uint64_t>(l) * small_per_layer,
                /*q_off=*/static_cast<uint64_t>(l) * q_per_layer,
                nb, nkh, vh, dv, dk, M, topk_q);
            !st.ok) {
            return scorer_backend_unavailable(
                failure_reason, "deltanet_scorer_kernel_failed", st);
        }
    }

    // Aggregate over heads (TopKMean_h), per-layer RMS-normalize over blocks, and
    // sum across layers with equal weights. Done on host (nb*vh*L floats).
    std::vector<float> r(small_per_layer * L, 0.0f);
    if (auto cp = backend_.copy_to_host(*g_deltanet_r_, r.data(), 0, r.size());
        !cp.ok) {
        return scorer_backend_unavailable(
            failure_reason, "deltanet_score_d2h_failed", cp);
    }
    std::vector<double> layer_block(static_cast<uint64_t>(L) * nb, 0.0);
    for (uint32_t l = 0; l < L; ++l) {
        for (uint32_t j = 0; j < nb; ++j) {
            // TopKMean over heads.
            std::vector<float> hv(vh);
            for (uint32_t h = 0; h < vh; ++h) {
                hv[h] = r[(static_cast<uint64_t>(l) * nb + j) * vh + h];
            }
            const uint32_t kh = std::min(topk_h, vh);
            std::partial_sort(hv.begin(), hv.begin() + kh, hv.end(),
                              std::greater<float>());
            double sum = 0.0;
            for (uint32_t i = 0; i < kh; ++i) sum += hv[i];
            layer_block[static_cast<uint64_t>(l) * nb + j] =
                kh > 0 ? sum / kh : 0.0;
        }
        // Per-layer RMS normalize across blocks.
        double ms = 0.0;
        for (uint32_t j = 0; j < nb; ++j) {
            const double v = layer_block[static_cast<uint64_t>(l) * nb + j];
            ms += v * v;
        }
        const double rms = std::sqrt(ms / std::max<uint32_t>(nb, 1)) + 1e-6;
        for (uint32_t j = 0; j < nb; ++j) {
            layer_block[static_cast<uint64_t>(l) * nb + j] /= rms;
        }
    }
    std::vector<double> best(block_store_->block_count(), 0.0);
    const double wl = 1.0 / static_cast<double>(L);
    for (uint32_t j = 0; j < nb && j < best.size(); ++j) {
        double s = 0.0;
        for (uint32_t l = 0; l < L; ++l) {
            s += wl * layer_block[static_cast<uint64_t>(l) * nb + j];
        }
        best[j] = s;
    }
    block_store_->set_retrieval_scores(best);
    if (std::getenv("QW3_KVMEM_TRACE") || std::getenv("QW3_KVMEM_DELTANET_TRACE")) {
        std::fprintf(stderr,
            "[bs-deltanet] scored %u blocks over L_dn=%u layers, M=%u query "
            "tokens, decay=%s\n",
            nb, L, M, bc.deltanet_decay ? "on" : "off");
    }
    return true;
}

void QwenExecutor::kvmem_drain_query_capture_slot(uint32_t slot_index) {
    if (slot_index >= g_query_capture_slots_.size()) return;
    QueryCaptureSlot &slot = g_query_capture_slots_[slot_index];
    if (!slot.pending) return;
    require_status(backend_.wait_kv_transfer_fence(slot.copy_done));
    if (!g_query_multi_host_ ||
        slot.dst_elem_offset > g_query_multi_host_capacity_ ||
        slot.elem_count >
            g_query_multi_host_capacity_ - slot.dst_elem_offset ||
        !slot.pinned ||
        slot.elem_count * sizeof(uint16_t) > slot.pinned->bytes) {
        throw std::runtime_error(
            "KVMem host query capture completed outside its destination");
    }
    std::memcpy(
        g_query_multi_host_.get() + slot.dst_elem_offset,
        slot.pinned->data,
        static_cast<size_t>(slot.elem_count) * sizeof(uint16_t));
    slot.pending = false;
    slot.dst_elem_offset = 0;
    slot.elem_count = 0;
}

void QwenExecutor::kvmem_drain_query_capture() {
    for (uint32_t i = 0; i < g_query_capture_slots_.size(); ++i) {
        kvmem_drain_query_capture_slot(i);
    }
}

// During prefill at bs_score_layer_: de-RoPE the in-span Q rows of the current
// chunk into g_query_multi_ at their true bake positions. The overlap of the
// chunk's absolute prompt-token indices [base_pos, base_pos+batch) with [qb,qe) is a
// contiguous run, so one batched launch covers it; counts accumulate across
// chunks until the whole span is captured. Inert unless a span is active.
void QwenExecutor::kvmem_capture_query_multi(uint32_t slot, uint32_t chunk_off,
                                             uint32_t batch, uint32_t base_pos,
                                             uint32_t rope_base_pos,
                                             uint32_t q_token_stride) {
    if (kvmem_query_end_ <= kvmem_query_begin_) return;
    if (!g_query_multi_ || !q_batch_ || g_query_multi_ready_) return;
    // base_pos and [query_begin,query_end) are both absolute positions in the
    // complete prompt. This remains true after prefix reuse: restore_state sets
    // position() to checkpoint C and suffix prefill begins at base_pos=C.
    (void)chunk_off;
    const QuerySpanOverlap overlap = query_span_overlap(
        base_pos, batch, kvmem_query_begin_, kvmem_query_end_);
    if (overlap.count == 0) return;
    const uint32_t r0 = overlap.row_offset;    // chunk-local first in-span row
    const uint32_t cnt = overlap.count;        // rows captured this chunk
    const QwenConfig &cfg = model_.config();
    const uint32_t n_heads = cfg.n_heads;
    const uint32_t head_dim = cfg.head_dim;
    const uint32_t q_head_stride = 2 * head_dim;  // attn-Q is first head_dim/unit
    const uint32_t S = kvmem_query_span_;         // per-layer row stride
    const uint64_t q_elem_off = static_cast<uint64_t>(r0) * q_token_stride;
    // De-RoPE at the rotation actually applied to this Q row. When the window is
    // active (long prompt past budget) Q was RoPE'd at the window position, not
    // the true bake position; rope_base_pos carries that (== base_pos otherwise),
    // so the captured query lands in the same content frame as the k̄ index.
    const int32_t start_pos = static_cast<int32_t>(rope_base_pos + r0);
    const int32_t actual_layer = slot < std_layers_.size()
        ? static_cast<int32_t>(std_layers_[slot]) : -1;
    trace_rope_position_if_out_of_range(
        "kvmem_capture_query_multi.derope", start_pos, cnt,
        cfg.n_ctx_train, actual_layer);
    const uint64_t row_elems =
        static_cast<uint64_t>(n_heads) * head_dim;
    if (kvmem_query_host_capture_) {
        const uint32_t capture_slot =
            g_query_capture_next_slot_++ %
            static_cast<uint32_t>(g_query_capture_slots_.size());
        kvmem_drain_query_capture_slot(capture_slot);
        QueryCaptureSlot &stage =
            g_query_capture_slots_[capture_slot];
        const uint64_t elems =
            static_cast<uint64_t>(cnt) * row_elems;
        if (!stage.device || stage.capacity_elems < elems) {
            stage.device =
                backend_.tensor_f16(elems, "g_query_capture_stage");
            stage.capacity_elems = elems;
        }
        const uint64_t bytes = elems * sizeof(uint16_t);
        if (!stage.pinned || stage.pinned->bytes < bytes) {
            stage.pinned =
                backend_.host_buffer(bytes, "g_query_capture_bounce");
        }
        auto st = backend_.derope_query_multi_device(
            *stage.device, *q_batch_, q_elem_off, /*out_elem_offset=*/0,
            q_token_stride, q_head_stride, cnt, n_heads, head_dim,
            cfg.rope_dim, start_pos, cfg.rope_theta);
        if (!st.ok) return;
        require_status(backend_.begin_kv_transfer_from_device());
        require_status(backend_.copy_bytes_to_host_async(
            *stage.device, stage.pinned->data, /*byte_offset=*/0, bytes));
        require_status(
            backend_.record_kv_transfer_fence(stage.copy_done));
        stage.dst_elem_offset =
            (static_cast<uint64_t>(slot) * S +
             g_query_multi_count_) * row_elems;
        stage.elem_count = elems;
        stage.pending = true;
    } else {
        // Legacy full-GPU layout: write this chunk into layer `slot`'s slice.
        const uint64_t out_elem_off =
            (static_cast<uint64_t>(slot) * S +
             g_query_multi_count_) * row_elems;
        auto st = backend_.derope_query_multi_device(
            *g_query_multi_, *q_batch_, q_elem_off, out_elem_off,
            q_token_stride, q_head_stride, cnt, n_heads, head_dim,
            cfg.rope_dim, start_pos, cfg.rope_theta);
        if (!st.ok) return;
    }
    // The shared per-slot token count tracks progress identically across all L
    // layers (every normal layer sees the same chunk in one forward pass), so
    // advance it ONCE per chunk — after the last slot has written its rows.
    const uint32_t L = std::max<uint32_t>(kvmem_qc_num_layers_, 1u);
    if (slot + 1 == L) {
        g_query_multi_count_ += cnt;
        if (g_query_multi_count_ >= S) {
            g_query_multi_ready_ = true;
        }
        if (std::getenv("QW3_KVMEM_TRACE")) {
            std::fprintf(stderr,
                "[bs-qcap] slot=%u rope_base=%u r0=%u cnt=%u count=%u/%u ready=%d\n",
                slot, rope_base_pos, r0, cnt, g_query_multi_count_, S,
                g_query_multi_ready_ ? 1 : 0);
        }
    }
}

// Clean-query prefill (task #50). After a PASS-A prefill of the question tokens in
// isolation, g_query_multi_ holds the recency-free de-RoPE'd query. Copy it into a
// persistent buffer that the reset_state between the two passes will not clear.
void QwenExecutor::kvmem_stash_clean_query() {
    if (!g_query_multi_ || g_query_multi_count_ == 0) return;
    const uint64_t n = g_query_multi_->count;
    if (!g_query_multi_clean_ || g_query_multi_clean_->count < n) {
        g_query_multi_clean_ = backend_.tensor_f32(n, "g_query_multi_clean");
    }
    require_status(backend_.copy_d2d(*g_query_multi_clean_, *g_query_multi_, 0, n));
    g_query_multi_clean_count_ = g_query_multi_count_;
    if (std::getenv("QW3_KVMEM_TRACE")) {
        std::fprintf(stderr, "[bs-cleanq] stash rows=%u elems=%llu\n",
                     g_query_multi_clean_count_,
                     static_cast<unsigned long long>(n));
    }
}

// Restore the stashed clean query into g_query_multi_ and mark it ready, so the
// (mid-prefill and decode-time) reselects rank blocks by the recency-free query.
// g_query_multi_ is re-allocated at the same [L,S,...] size in PASS B, so the copy
// count matches; clamp to the smaller of the two just in case.
void QwenExecutor::kvmem_restore_clean_query() {
    if (!g_query_multi_clean_ || g_query_multi_clean_count_ == 0 || !g_query_multi_)
        return;
    const uint64_t n =
        std::min<uint64_t>(g_query_multi_->count, g_query_multi_clean_->count);
    require_status(backend_.copy_d2d(*g_query_multi_, *g_query_multi_clean_, 0, n));
    g_query_multi_count_ = g_query_multi_clean_count_;
    g_query_multi_ready_ = true;
    if (std::getenv("QW3_KVMEM_TRACE")) {
        std::fprintf(stderr, "[bs-cleanq] restore rows=%u\n",
                     g_query_multi_count_);
    }
}

// Query replay requires its suffix [pin_from, block_count()) to survive the
// semantic selection. These blocks consume ordinary top-k slots: appending them
// after a full top-k would grow the active window beyond select_budget and could
// drive target/MTP RoPE past n_ctx_train near the end of the generation reserve.
// With no pin (the default cleared by reset_state), preserve the ordinary
// selector byte-for-byte.
void QwenExecutor::kvmem_set_oracle_token_spans(
        const std::vector<std::pair<uint32_t, uint32_t>> &spans,
        bool oracle_only) {
    kvmem_oracle_token_spans_.clear();
    kvmem_oracle_only_ = oracle_only && !spans.empty();
    kvmem_oracle_token_spans_.reserve(spans.size());
    for (const auto &[begin, end] : spans) {
        if (end <= begin) {
            throw std::runtime_error(
                "KVMem oracle token spans must be non-empty");
        }
        kvmem_oracle_token_spans_.emplace_back(begin, end);
    }
    if (!kvmem_oracle_token_spans_.empty() &&
        std::getenv("QW3_KVMEM_TRACE")) {
        std::fprintf(stderr, "[bs-oracle] configured token_spans=%zu\n",
                     kvmem_oracle_token_spans_.size());
    }
}

std::vector<uint32_t> QwenExecutor::kvmem_selection_with_pin() {
    if (kvmem_qc_pin_from_block_ == 0xffffffffu &&
        kvmem_oracle_token_spans_.empty()) {
        return block_store_->pick_topk_blocks();
    }
    const uint32_t nb = block_store_->block_count();
    std::vector<uint32_t> mandatory;
    if (kvmem_qc_pin_from_block_ < nb) {
        mandatory.reserve(nb - kvmem_qc_pin_from_block_);
        for (uint32_t id = kvmem_qc_pin_from_block_; id < nb; ++id) {
            mandatory.push_back(id);
        }
    }
    if (!kvmem_oracle_token_spans_.empty()) {
        const uint32_t bt =
            std::max<uint32_t>(1, block_store_->config().block_tokens);
        for (const auto &[begin, end] : kvmem_oracle_token_spans_) {
            const uint32_t first = begin / bt;
            const uint32_t last = (end - 1) / bt;
            if (first >= nb) continue;
            const uint32_t capped_last = std::min(last, nb - 1);
            for (uint32_t id = first; id <= capped_last; ++id) {
                mandatory.push_back(id);
            }
        }
        std::sort(mandatory.begin(), mandatory.end());
        mandatory.erase(std::unique(mandatory.begin(), mandatory.end()),
                        mandatory.end());
        if (std::getenv("QW3_KVMEM_TRACE")) {
            std::fprintf(stderr,
                         "[bs-oracle] materialized_blocks=%u mandatory=%zu\n",
                         nb, mandatory.size());
        }
    }
    if (kvmem_oracle_only_) {
        const uint32_t sink =
            std::min(block_store_->config().sink_blocks, nb);
        for (uint32_t id = 0; id < sink; ++id) {
            mandatory.push_back(id);
        }
        std::sort(mandatory.begin(), mandatory.end());
        mandatory.erase(std::unique(mandatory.begin(), mandatory.end()),
                        mandatory.end());
        if (mandatory.size() > block_store_->budget_blocks()) {
            throw std::runtime_error(
                "KVMem oracle-only selection exceeds the configured budget");
        }
        if (std::getenv("QW3_KVMEM_TRACE")) {
            std::fprintf(stderr,
                         "[bs-oracle] mode=only selected=%zu sink=%u\n",
                         mandatory.size(), sink);
        }
        return mandatory;
    }
    return block_store_->pick_topk_blocks(mandatory);
}

// During prefill at every normal-attention layer: build the per-layer content
// mean-key for the chunk's blocks directly from the freshly-RoPE'd K batch (#91).
// Unlike kvmem_capture_query_multi (which captures only the in-span QUESTION rows)
// this captures EVERY block, so the full index covers the whole history regardless
// of later GPU-pool offload / window re-RoPE. The chunk's tokens [base_pos,
// base_pos+batch) are block-aligned (chunk_size % block_tokens == 0, fresh request
// => base_pos % block_tokens == 0), so chunk-local block j owns rows [j*bt,
// (j+1)*bt) and lands at global block first_block+j. K was RoPE-baked at
// rope_base_pos (the window position when the window is active, the true position
// otherwise); de-RoPE there so the content frame matches the query capture.
// Captured-block progress advances once per chunk (after the last slot writes);
// when the whole history is covered the index is published (g_kbar_/g_kbar_multi_
// ready, g_content_ready_), which also keeps the one-shot paged builder gated off.
void QwenExecutor::kvmem_capture_kbar_multi(uint32_t slot, uint32_t batch,
                                            uint32_t base_pos,
                                            uint32_t rope_base_pos,
                                            uint32_t k_token_stride) {
    if (!kvmem_qc_capture_active_) return;
    if (!g_kbar_multi_ || !k_batch_ || g_kbar_multi_ready_) return;
    if (kvmem_qc_total_blocks_ == 0 || !block_store_) return;
    const QwenConfig &cfg = model_.config();
    const uint32_t n_kv_heads = cfg.n_kv_heads;
    const uint32_t head_dim = cfg.head_dim;
    const uint32_t bt = std::max<uint32_t>(block_store_->config().block_tokens, 1u);
    // Per-layer slice base uses the FIXED session stride (ctx_blocks) when set, so
    // preserved [0,D) slices stay put as the block count grows across resumed turns
    // (server-side session continuation). Falls back to the turn's block count when
    // no fixed stride was pinned (byte-identical legacy layout).
    const uint32_t stride_blocks =
        kvmem_qc_layer_stride_blocks_ != 0 ? kvmem_qc_layer_stride_blocks_
                                           : kvmem_qc_total_blocks_;
    // Every write must extend the exact contiguous prefix represented by the
    // index. All standard layers see the same batch; only the last slot advances
    // this cursor below. A block-aligned replay rewinds the cursor explicitly in
    // kvmem_begin_query_replay and overwrites the suffix from that boundary.
    if (base_pos != kvmem_qc_captured_tokens_) {
        throw std::runtime_error(
            "KVMem incremental mean-K capture is not contiguous");
    }
    const uint32_t off = base_pos % bt;
    const uint32_t first_block = base_pos / bt;
    if (first_block >= kvmem_qc_total_blocks_) return;
    const uint32_t n_blocks_chunk = std::min(
        (off + batch + bt - 1) / bt,
        kvmem_qc_total_blocks_ - first_block);
    if (n_blocks_chunk == 0) return;
    const uint64_t per_pos = static_cast<uint64_t>(n_kv_heads) * head_dim;
    const uint64_t kbar_block_base =
        static_cast<uint64_t>(slot) * stride_blocks + first_block;
    DeviceStatus st;
    const int32_t actual_layer = slot < std_layers_.size()
        ? static_cast<int32_t>(std_layers_[slot]) : -1;
    trace_rope_position_if_out_of_range(
        "kvmem_capture_kbar_multi.derope", rope_base_pos, batch,
        cfg.n_ctx_train, actual_layer);
    if (off == 0) {
        // Preserve the original aligned kernel. Besides keeping ordinary cold
        // prefill byte-identical, overwrite semantics are exactly what a replayed
        // suffix needs after rollback.
        st = backend_.block_kmean_content_batch_device(
            *k_batch_, *g_kbar_multi_, kbar_block_base, n_blocks_chunk,
            k_token_stride, batch, bt, n_kv_heads, head_dim, cfg.rope_dim,
            static_cast<int32_t>(rope_base_pos), cfg.rope_theta,
            /*src_row_off=*/0, /*n_subblocks=*/kvmem_qc_n_subblocks_);
    } else {
        st = backend_.block_kmean_content_batch_merge_device(
            *k_batch_, *g_kbar_multi_, kbar_block_base, n_blocks_chunk,
            k_token_stride, batch, bt, off, n_kv_heads, head_dim,
            cfg.rope_dim, static_cast<int32_t>(rope_base_pos), cfg.rope_theta,
            /*n_subblocks=*/kvmem_qc_n_subblocks_);
    }
    if (!st.ok) return;
    // Raw-key MaxSim (#104): in ADDITION to the mean, store each chunk row's
    // de-RoPE'd K into g_kraw_multi_ so the scorer can MAX over a block's tokens.
    // Chunk row r is global token (base_pos + r); store [base_pos, base_pos+rows)
    // capped at total_tokens. out_base_elem indexes (slot, base_pos) in fp32 elems.
    if (kvmem_qc_pertoken_ && g_kraw_multi_ &&
        kvmem_qc_total_tokens_ > 0 && base_pos < kvmem_qc_total_tokens_) {
        const uint32_t store_rows =
            std::min(batch, kvmem_qc_total_tokens_ - base_pos);
        const uint64_t out_base_elem =
            (static_cast<uint64_t>(slot) * kvmem_qc_total_tokens_ + base_pos) *
            n_kv_heads * head_dim;
        trace_rope_position_if_out_of_range(
            "kvmem_capture_kbar_multi.raw_derope", rope_base_pos,
            store_rows, cfg.n_ctx_train, actual_layer);
        (void)backend_.derope_store_content_batch_device(
            *k_batch_, *g_kraw_multi_, out_base_elem, k_token_stride, store_rows,
            n_kv_heads, head_dim, cfg.rope_dim,
            static_cast<int32_t>(rope_base_pos), cfg.rope_theta);
    }
    // Advance the exact token cursor once, after every layer has updated the same
    // destination range. captured_blocks remains a diagnostic ceil(prefix/bt).
    const uint32_t L = std::max<uint32_t>(kvmem_qc_num_layers_, 1u);
    if (slot + 1 != L) return;
    kvmem_qc_captured_tokens_ = base_pos + batch;
    kvmem_qc_captured_blocks_ =
        (kvmem_qc_captured_tokens_ + bt - 1) / bt;
    if (kvmem_qc_captured_tokens_ < kvmem_qc_prompt_tokens_) return;
    // Whole history covered: publish the index. Copy the slot-0 slice into g_kbar_
    // so the single last-token fallback scorer ranks against the same full-coverage
    // index. Mark the global index live so the one-shot paged builder stays gated
    // off at the reselect boundary.
    // The single last-token fallback index (g_kbar_) is laid out one mean per
    // block; only seed it from slot 0 when g_kbar_multi_ shares that layout
    // (n_subblocks == 1). Under sub-block mode the leading per_layer elements are
    // a block's sub-block means, not a contiguous block run, so the D2D copy would
    // mis-seed the fallback — skip it (the fallback scorer is unavailable in
    // sub-block mode, guarded at the shmem-cap check).
    const uint64_t per_layer =
        static_cast<uint64_t>(kvmem_qc_total_blocks_) * per_pos;
    if (kvmem_qc_n_subblocks_ == 1 && g_kbar_ && g_kbar_->count >= per_layer) {
        (void)backend_.copy_d2d(*g_kbar_, *g_kbar_multi_, /*src_offset=*/0,
                                per_layer);
    }
    g_content_ready_ = true;
    g_indexed_blocks_ = kvmem_qc_total_blocks_;
    g_kbar_multi_ready_ = true;
    g_kbar_multi_blocks_ = kvmem_qc_total_blocks_;
    if (kvmem_qc_pertoken_ && g_kraw_multi_) {
        g_kraw_multi_ready_ = true;
    }
    if (std::getenv("QW3_KVMEM_TRACE")) {
        std::fprintf(stderr,
            "[bs-kbar-build] incremental index complete: L=%u blocks=%u "
            "(slot0->g_kbar)\n",
            L, kvmem_qc_total_blocks_);
    }
}

// Enable decode-time content capture for the coming (plain) decode loop. Only
// arms when kvmem + a fixed-stride QC index are live (above-budget mean-k turn);
// below budget / off / per-token it is a no-op, so the caller can invoke it
// unconditionally. Resets the per-block staging cursor so the first generated
// block boundary starts fresh.
void QwenExecutor::kvmem_decode_capture_begin() {
    kvmem_decode_capture_on_ = false;
    decode_stage_active_ = false;
    decode_stage_rows_ = 0;
    decode_stage_block_ = 0;
    if (!kvmem_enabled_ || !block_store_) return;
    if (!g_kbar_multi_ || kvmem_qc_layer_stride_blocks_ == 0) return;
    if (kvmem_qc_pertoken_) return;                       // per-token index not fixed-stride
    if (kvmem_qc_n_subblocks_ > 1) return;                // sub-block index: decode capture is a follow-up
    if (kvmem_query_end_ <= kvmem_query_begin_) return;   // no QC span -> below budget / dense
    kvmem_decode_capture_on_ = true;
}

// Per-std-layer decode hook (inside forward_one_token, after K RoPE, before the
// KV append). Stages the current token's post-RoPE K row (k_) de-RoPE'd at its
// actual bake position rope_pos, so the staged row is in the position-invariant
// content frame (immune to later reselect re-baking). A block is staged only when
// its FIRST (offset-0) token was itself generated; the prompt/response boundary
// block (whose offset-0 token is a prompt token) is left with its prefill partial
// mean (a bounded ranking approximation; KV bytes stay exact). On the last std
// layer of a token that completes a block, the block is meaned into g_kbar_multi_.
void QwenExecutor::kvmem_decode_capture_stage(uint32_t layer_index,
                                              uint32_t rope_pos) {
    if (!kvmem_decode_capture_on_ || !k_ || !g_kbar_multi_) return;
    const int32_t slot = (static_cast<size_t>(layer_index) < std_layer_slot_.size())
                             ? std_layer_slot_[layer_index] : -1;
    if (slot < 0) return;
    const QwenConfig &cfg = model_.config();
    const uint32_t n_kv_heads = cfg.n_kv_heads;
    const uint32_t head_dim = cfg.head_dim;
    const uint32_t per_pos = n_kv_heads * head_dim;
    const uint32_t bt = std::max<uint32_t>(block_store_->config().block_tokens, 1u);
    const uint32_t L = std::max<uint32_t>(kvmem_qc_num_layers_, 1u);
    const uint32_t p = position_;                 // true position of this token
    const uint32_t i = p % bt;                     // row within its block
    const uint32_t block = p / bt;
    // Decide block membership once per token (on the first std layer). A block is
    // eligible only if we captured its offset-0 token: on i==0 start a fresh block;
    // otherwise continue only if we are already mid-staging THIS block.
    if (slot == 0) {
        if (i == 0) {
            decode_stage_active_ = true;
            decode_stage_block_ = block;
        } else if (!(decode_stage_active_ && decode_stage_block_ == block)) {
            decode_stage_active_ = false;
        }
    }
    if (!decode_stage_active_ || decode_stage_block_ != block) return;
    if (block >= kvmem_qc_layer_stride_blocks_) {         // beyond the fixed index
        decode_stage_active_ = false;
        return;
    }
    // Lazily size the content-frame staging buffer [L, block_tokens, per_pos].
    const uint64_t stage_rows = static_cast<uint64_t>(L) * bt;
    if (!g_kbar_decode_stage_ ||
        g_kbar_decode_stage_->count < stage_rows * per_pos) {
        g_kbar_decode_stage_ =
            backend_.tensor_f32(stage_rows * per_pos, "g_kbar_decode_stage");
    }
    // De-RoPE the current layer's K (single row, baked at rope_pos) into the
    // content frame and stage it at slot-row i. k_ is [n_kv_heads, head_dim].
    const uint64_t out_base =
        (static_cast<uint64_t>(slot) * bt + i) * per_pos;
    trace_rope_position_if_out_of_range(
        "kvmem_decode_capture_stage.derope", rope_pos, 1,
        cfg.n_ctx_train, static_cast<int32_t>(layer_index));
    (void)backend_.derope_store_content_batch_device(
        *k_, *g_kbar_decode_stage_, out_base, /*k_stride=*/per_pos, /*batch=*/1,
        n_kv_heads, head_dim, cfg.rope_dim, static_cast<int32_t>(rope_pos),
        cfg.rope_theta);
    // Advance the row cursor + finalize a completed block once per token (after
    // the last std layer has staged its slice; progress is identical across L).
    if (static_cast<uint32_t>(slot) + 1 == L) {
        decode_stage_rows_ = i + 1;
        if (i + 1 == bt) {
            kvmem_capture_decode_block(block, bt);
            decode_stage_active_ = false;
            decode_stage_rows_ = 0;
        }
    }
}

// Mean the `rows` staged content-frame rows of the just-completed decode block
// into g_kbar_multi_[slot*stride + true_block_index] for every std layer. The
// staged rows are ALREADY de-RoPE'd (per-token, at capture time), so the mean
// applies NO further rotation (rope_dim==0); src_row_off selects slot s's rows.
// Writes only the index BUFFER — the block/coverage counts that drive THIS turn's
// scorer are left untouched (a reselect mid-decode must keep scoring the prompt's
// blocks); the NEXT turn re-establishes coverage via kvmem_set_query_span's resume
// seeding, which trusts these preserved slices.
void QwenExecutor::kvmem_capture_decode_block(uint32_t true_block_index,
                                              uint32_t rows) {
    if (!g_kbar_decode_stage_ || !g_kbar_multi_ || rows == 0) return;
    if (true_block_index >= kvmem_qc_layer_stride_blocks_) return;
    const QwenConfig &cfg = model_.config();
    const uint32_t n_kv_heads = cfg.n_kv_heads;
    const uint32_t head_dim = cfg.head_dim;
    const uint32_t per_pos = n_kv_heads * head_dim;
    const uint32_t bt = std::max<uint32_t>(block_store_->config().block_tokens, 1u);
    const uint32_t L = std::max<uint32_t>(kvmem_qc_num_layers_, 1u);
    const uint32_t stride = kvmem_qc_layer_stride_blocks_;
    for (uint32_t s = 0; s < L; ++s) {
        const uint64_t kbar_block_base =
            static_cast<uint64_t>(s) * stride + true_block_index;
        (void)backend_.block_kmean_content_batch_device(
            *g_kbar_decode_stage_, *g_kbar_multi_, kbar_block_base,
            /*n_blocks_chunk=*/1, /*k_stride=*/per_pos, /*batch=*/rows,
            /*blk_tokens=*/bt, n_kv_heads, head_dim, /*rope_dim=*/0,
            /*rope_base=*/0, cfg.rope_theta, /*src_row_off=*/s * bt);
    }
    if (std::getenv("QW3_KVMEM_TRACE")) {
        std::fprintf(stderr,
            "[bs-decode-cap] block=%u rows=%u -> g_kbar_multi_ (L=%u stride=%u)\n",
            true_block_index, rows, L, stride);
    }
}

// End-of-turn: finalize a trailing partial decode block (a response whose last
// block is < block_tokens long) so it carries a mean for the next turn's resume
// seeding, then disarm capture. Idempotent; safe to call when capture is off.
void QwenExecutor::kvmem_decode_capture_finalize() {
    if (kvmem_decode_capture_on_ && decode_stage_active_ && decode_stage_rows_ > 0) {
        kvmem_capture_decode_block(decode_stage_block_, decode_stage_rows_);
    }
    kvmem_decode_capture_on_ = false;
    decode_stage_active_ = false;
    decode_stage_rows_ = 0;
    decode_stage_block_ = 0;
}

bool QwenExecutor::kvmem_score_host_query_chunks(
        uint32_t n_blocks, uint32_t kbar_stride, float scale,
        uint32_t excl_lo_end, uint32_t excl_hi_begin,
        std::string *failure_reason) {
    if (!kvmem_query_host_capture_ || !g_query_multi_host_ ||
        !g_query_multi_ || !g_kbar_multi_ || !g_score_dev_) {
        return scorer_unavailable(
            failure_reason, "host_query_stream_unavailable");
    }
    kvmem_drain_query_capture();
    const QwenConfig &cfg = model_.config();
    const uint32_t layers = kvmem_qc_num_layers_;
    const uint32_t tokens = g_query_multi_count_;
    const uint32_t source_stride = kvmem_query_span_;
    if (layers == 0 || tokens == 0 || tokens > source_stride) {
        return scorer_unavailable(
            failure_reason, "host_query_stream_shape_invalid");
    }
    const uint32_t chunk_tokens = std::min<uint32_t>(
        tokens, env_uint32_or("QW3_KVMEM_QUERY_SCORE_CHUNK", 256));
    const uint64_t row_elems =
        static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim;
    const uint64_t max_stage_elems =
        static_cast<uint64_t>(layers) * chunk_tokens * row_elems;
    if (g_query_multi_->elem_size != sizeof(uint16_t) ||
        g_query_multi_->count < max_stage_elems) {
        g_query_multi_ = backend_.tensor_f16(
            max_stage_elems, "g_query_score_stage");
        g_query_multi_capacity_ =
            static_cast<uint64_t>(layers) * chunk_tokens;
    }
    const uint64_t max_stage_bytes =
        max_stage_elems * sizeof(uint16_t);
    if (!g_query_score_pinned_ ||
        g_query_score_pinned_capacity_ < max_stage_bytes) {
        g_query_score_pinned_ =
            backend_.host_buffer(max_stage_bytes, "g_query_score_bounce");
        g_query_score_pinned_capacity_ = max_stage_bytes;
    }

    uint64_t gather_ns = 0;
    uint64_t h2d_wait_ns = 0;
    uint64_t copied_bytes = 0;
    uint32_t chunks = 0;
    const uint64_t total_begin = kvmem_steady_ns();
    std::unique_ptr<DeviceTransferFence> prior_score_done;
    for (uint32_t token_base = 0; token_base < tokens;) {
        const uint32_t count =
            std::min<uint32_t>(chunk_tokens, tokens - token_base);
        const uint64_t chunk_layer_elems =
            static_cast<uint64_t>(count) * row_elems;
        const uint64_t chunk_elems =
            static_cast<uint64_t>(layers) * chunk_layer_elems;
        const uint64_t chunk_bytes =
            chunk_elems * sizeof(uint16_t);

        const uint64_t gather_begin = kvmem_steady_ns();
        auto *packed =
            static_cast<uint16_t *>(g_query_score_pinned_->data);
        for (uint32_t layer = 0; layer < layers; ++layer) {
            const uint64_t src =
                (static_cast<uint64_t>(layer) * source_stride +
                 token_base) * row_elems;
            const uint64_t dst =
                static_cast<uint64_t>(layer) * chunk_layer_elems;
            std::memcpy(
                packed + dst, g_query_multi_host_.get() + src,
                static_cast<size_t>(chunk_layer_elems) *
                    sizeof(uint16_t));
        }
        gather_ns += kvmem_steady_ns() - gather_begin;

        // CPU packing of chunk N overlaps the exec-stream scoring of N-1.
        // Before H2D overwrites the single device staging tensor, make the copy
        // stream wait on the previous score's completion event.
        if (prior_score_done) {
            require_status(
                backend_.kv_transfer_wait_for_execution(prior_score_done));
        }
        require_status(backend_.begin_kv_transfer_to_device());
        require_status(backend_.copy_bytes_from_host_async(
            *g_query_multi_, /*byte_offset=*/0,
            g_query_score_pinned_->data, chunk_bytes));
        std::unique_ptr<DeviceTransferFence> h2d_done;
        require_status(
            backend_.record_kv_transfer_fence(h2d_done));
        require_status(
            backend_.execution_wait_for_kv_transfer(h2d_done));

        if (auto st = backend_.block_attn_score_softmax_pages_device(
                *g_score_dev_, *g_query_multi_, *g_kbar_multi_,
                layers, count, /*q_layer_stride=*/count, n_blocks,
                kbar_stride, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
                scale, excl_lo_end, excl_hi_begin,
                /*n_subblocks=*/kvmem_qc_n_subblocks_,
                /*reduce_max=*/kvmem_qc_subblock_max_ ? 1u : 0u,
                /*accumulate=*/chunks == 0 ? 0u : 1u);
            !st.ok) {
            return scorer_backend_unavailable(
                failure_reason, "host_query_chunk_scorer_failed", st);
        }
        require_status(
            backend_.record_execution_fence(prior_score_done));

        // The scorer may continue asynchronously, but the H2D source slab can
        // be repacked only after its copy has completed.
        const uint64_t wait_begin = kvmem_steady_ns();
        require_status(
            backend_.wait_kv_transfer_fence(h2d_done));
        h2d_wait_ns += kvmem_steady_ns() - wait_begin;
        copied_bytes += chunk_bytes;
        token_base += count;
        ++chunks;
    }
    const uint64_t total_ns = kvmem_steady_ns() - total_begin;
    if (std::getenv("QW3_KVMEM_TRACE") ||
        kvmem_perf_trace_flag()) {
        std::fprintf(
            stderr,
            "[bs-qscore-host] dtype=fp16 layers=%u tokens=%u "
            "chunk_tokens=%u chunks=%u copied_mib=%.2f "
            "gather_ms=%.3f h2d_wait_ms=%.3f total_submit_ms=%.3f\n",
            layers, tokens, chunk_tokens, chunks,
            static_cast<double>(copied_bytes) / (1024.0 * 1024.0),
            gather_ns / 1.0e6, h2d_wait_ns / 1.0e6,
            total_ns / 1.0e6);
    }
    return true;
}

// Query-conditioned MEAN-K scoring at the reselect boundary (--kvmem-retrieval-method
// mean-k, default): per (layer, question token, query head) softmax the q·k̄ dot
// products OVER PAGES so blocks compete, then accumulate that attention mass over the
// query tokens. Uses the cheap per-block mean-key buffer (g_kbar_multi_, ~28 MB), one
// fused launch + one D2H. The kernel's softmax IS the block distribution, so there is
// no host-side inv_lm divide or extra normalization tail. Returns false (caller falls
// back to the single last-token scorer) if the index/query isn't live or the kernel
// rejects the page count. See docs/kvmem_utility_eval_plan.md.
bool QwenExecutor::kvmem_retrieval_score_mean_softmax(
        int mask_mode, std::string *failure_reason) {
    if (failure_reason) failure_reason->clear();
    if (!block_store_) {
        return scorer_unavailable(failure_reason, "block_store_unavailable");
    }
    if (kvmem_query_end_ <= kvmem_query_begin_ ||
        kvmem_query_span_ == 0) {
        return scorer_unavailable(failure_reason, "query_span_missing");
    }
    if (!g_kbar_multi_ready_) {
        return scorer_unavailable(failure_reason,
                                  "mean_k_index_not_ready");
    }
    if (g_kbar_multi_blocks_ == 0) {
        return scorer_unavailable(failure_reason,
                                  "mean_k_index_has_zero_blocks");
    }
    if (!g_query_multi_) {
        return scorer_unavailable(failure_reason,
                                  "query_buffer_unavailable");
    }
    if (!g_query_multi_ready_ ||
        g_query_multi_count_ < kvmem_query_span_) {
        return scorer_unavailable(failure_reason,
                                  "query_capture_incomplete");
    }
    if (g_query_multi_count_ == 0) {
        return scorer_unavailable(failure_reason,
                                  "query_capture_empty");
    }
    if (!g_score_dev_) {
        return scorer_unavailable(failure_reason,
                                  "score_buffer_unavailable");
    }
    const QwenConfig &cfg = model_.config();
    const uint32_t L = kvmem_qc_num_layers_;
    const uint32_t M = g_query_multi_count_;
    const uint32_t S = kvmem_query_span_;       // per-layer row stride in g_query_multi_
    const uint32_t nb = g_kbar_multi_blocks_;
    if (L == 0) {
        return scorer_unavailable(failure_reason,
                                  "mean_k_has_zero_layers");
    }
    if (nb > g_kbar_global_capacity_) {
        return scorer_unavailable(failure_reason,
                                  "indexed_blocks_exceed_score_capacity");
    }

    // Per-(layer, question token, head) softmax over pages: score[w] comes back
    // already equal to the accumulated attention mass each page receives, so there
    // is no inv_lm divide and no extra normalization tail (the kernel's softmax IS
    // the page distribution). The CUDA backend preserves this exact computation
    // above the fused kernel's shared-memory cap via a bounded one-dot path.
    // The former two-dot implementation remains available as an explicit
    // performance/correctness A/B baseline.
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
    // Mask the always-kept sink [0,sink) and recent [nb-recent,nb) bands out of
    // the softmax so their probability mass (dominated by the boundary/adjacency
    // artifact) redistributes onto the retrievable middle. pick_topk_blocks keeps
    // those bands unconditionally regardless of score, so zeroing them here never
    // drops a kept block; it only sharpens the ranking of the contested middle.
    // Only meaningful over budget (nb>budget) where selection is actually
    // competitive. Default ON; QW3_KVMEM_MASK_KEPT=0 restores the un-masked path.
    uint32_t excl_lo_end = 0;
    uint32_t excl_hi_begin = UINT32_MAX;
    {
        bool mask_on;
        if (mask_mode == 0) {
            mask_on = false;
        } else if (mask_mode == 1) {
            mask_on = true;
        } else {
            const char *mask_env = std::getenv("QW3_KVMEM_MASK_KEPT");
            mask_on = !(mask_env && std::string(mask_env) == "0");
        }
        const uint32_t budget = block_store_->budget_blocks();
        if (mask_on && budget > 0 && nb > budget) {
            const KvMemStoreConfig &bcfg = block_store_->config();
            uint32_t sink = std::min(bcfg.sink_blocks, nb);
            const uint32_t recent = std::min(bcfg.recent_blocks, nb);
            // Only mask if a non-empty retrievable middle survives.
            if (sink + recent < nb) {
                excl_lo_end = sink;
                excl_hi_begin = nb - recent;
                if (std::getenv("QW3_KVMEM_TRACE")) {
                    std::fprintf(stderr,
                                 "[bs-mean-softmax] masking kept bands: "
                                 "sink[0,%u) recent[%u,%u) middle=%u blocks\n",
                                 excl_lo_end, excl_hi_begin, nb,
                                 excl_hi_begin - excl_lo_end);
                }
            }
        }
    }
    // kbar_layer_stride is the fixed session stride when resumable indexing is live
    // (g_kbar_multi_ over-allocated at ctx_blocks so preserved [0,D) slices survive
    // a growing block count); otherwise it equals nb (byte-identical legacy layout).
    const uint32_t kbar_stride =
        kvmem_qc_layer_stride_blocks_ != 0 ? kvmem_qc_layer_stride_blocks_ : nb;
    if (kvmem_query_host_capture_) {
        if (!kvmem_score_host_query_chunks(
                nb, kbar_stride, sm_scale, excl_lo_end,
                excl_hi_begin, failure_reason)) {
            return false;
        }
    } else {
        if (auto st = backend_.block_attn_score_softmax_pages_device(
                *g_score_dev_, *g_query_multi_, *g_kbar_multi_,
                L, M, /*q_layer_stride=*/S, nb,
                /*kbar_layer_stride=*/kbar_stride,
                cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, sm_scale,
                excl_lo_end, excl_hi_begin,
                /*n_subblocks=*/kvmem_qc_n_subblocks_,
                /*reduce_max=*/kvmem_qc_subblock_max_ ? 1u : 0u,
                /*accumulate=*/0);
            !st.ok) {
            return scorer_backend_unavailable(
                failure_reason, "mean_k_scorer_kernel_failed", st);
        }
    }
    std::vector<float> score(nb, 0.0f);
    if (auto cp = backend_.copy_to_host(*g_score_dev_, score.data(), 0, nb);
        !cp.ok) {
        return scorer_backend_unavailable(
            failure_reason, "mean_k_score_d2h_failed", cp);
    }
    std::vector<double> best(block_store_->block_count(), 0.0);
    for (uint32_t id = 0; id < nb && id < best.size(); ++id) {
        best[id] = static_cast<double>(score[id]);
    }
    block_store_->set_retrieval_scores(best);
    if (std::getenv("QW3_KVMEM_TRACE")) {
        std::fprintf(stderr,
                     "[bs-mean-softmax] query-conditioned softmax-pages: "
                     "L=%u M=%u tokens scored %u blocks\n",
                     L, M, nb);
    }
    return true;
}

// Raw-key ExactMass scoring at the reselect boundary (AgentKV port): instead of
// the diluting mean key (q·k̄ ≡ mean_t(q·k)), softmax the per-token logit
// scale·(q·k_raw) over ALL key tokens and sum each block's attention MASS
// (Σ_{t∈w} softmax_t). exp is applied to every raw token THEN summed (Σexp under
// one global denominator), so a needle token's mass survives AND blocks compete
// globally — the faithful analog of AgentKV's _stream_exact_mass_scores_pagewise
// (vs the inert softmax-over-pages path, which exps the already-diluted per-block
// mean). The kernel's softmax IS the block distribution, so there is no extra
// normalization tail; the masses (mean over L·M·n_heads of each block's softmax
// mass) are set as the scores directly. Returns false (caller falls back) if the
// raw-key buffer / query isn't live or the kernel rejects nb (shmem cap).
bool QwenExecutor::kvmem_retrieval_score_exactmass(
        std::string *failure_reason) {
    if (failure_reason) failure_reason->clear();
    if (!block_store_) {
        return scorer_unavailable(failure_reason, "block_store_unavailable");
    }
    if (kvmem_query_end_ <= kvmem_query_begin_ ||
        kvmem_query_span_ == 0) {
        return scorer_unavailable(failure_reason, "query_span_missing");
    }
    if (!g_kraw_multi_) {
        return scorer_unavailable(failure_reason,
                                  "exactmass_raw_key_buffer_unavailable");
    }
    if (!g_kraw_multi_ready_) {
        return scorer_unavailable(failure_reason,
                                  "exactmass_raw_key_index_not_ready");
    }
    if (kvmem_qc_total_tokens_ == 0) {
        return scorer_unavailable(failure_reason,
                                  "exactmass_index_has_zero_tokens");
    }
    if (kvmem_qc_total_blocks_ == 0) {
        return scorer_unavailable(failure_reason,
                                  "exactmass_index_has_zero_blocks");
    }
    if (!g_query_multi_) {
        return scorer_unavailable(failure_reason,
                                  "query_buffer_unavailable");
    }
    if (!g_query_multi_ready_ ||
        g_query_multi_count_ < kvmem_query_span_) {
        return scorer_unavailable(failure_reason,
                                  "query_capture_incomplete");
    }
    if (g_query_multi_count_ == 0) {
        return scorer_unavailable(failure_reason,
                                  "query_capture_empty");
    }
    if (!g_score_dev_) {
        return scorer_unavailable(failure_reason,
                                  "score_buffer_unavailable");
    }
    const QwenConfig &cfg = model_.config();
    const uint32_t L = kvmem_qc_num_layers_;
    const uint32_t M = g_query_multi_count_;
    const uint32_t S = kvmem_query_span_;       // per-layer row stride in g_query_multi_
    const uint32_t nb = kvmem_qc_total_blocks_;
    const uint32_t total_tokens = kvmem_qc_total_tokens_;
    if (L == 0) {
        return scorer_unavailable(failure_reason,
                                  "exactmass_has_zero_layers");
    }
    if (nb > g_kbar_global_capacity_) {
        return scorer_unavailable(failure_reason,
                                  "indexed_blocks_exceed_score_capacity");
    }
    const uint32_t bt =
        std::max<uint32_t>(block_store_->config().block_tokens, 1u);

    if (auto st = backend_.zero_tensor(*g_score_dev_); !st.ok) {
        return scorer_backend_unavailable(
            failure_reason, "score_buffer_zero_failed", st);
    }

    // scale = 1/sqrt(head_dim) (true softmax temperature). Each CTA emits a softmax
    // distribution over blocks; head_w averages over the per-(layer, query token,
    // head) CTAs so the final score is the MEAN attention mass each block receives.
    const uint32_t eff_heads = cfg.n_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
    const float head_w =
        1.0f / (static_cast<float>(L) * static_cast<float>(M) *
                static_cast<float>(eff_heads));
    const uint64_t q_layer_elems =
        static_cast<uint64_t>(S) * cfg.n_heads * cfg.head_dim;
    const uint64_t kraw_layer_elems =
        static_cast<uint64_t>(total_tokens) * cfg.n_kv_heads * cfg.head_dim;
    for (uint32_t slot = 0; slot < L; ++slot) {
        if (auto st = backend_.block_attn_score_exactmass_device(
                *g_score_dev_, *g_query_multi_, *g_kraw_multi_, M, total_tokens,
                nb, bt, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, scale, head_w,
                /*q_elem_off=*/static_cast<uint64_t>(slot) * q_layer_elems,
                /*kraw_elem_off=*/static_cast<uint64_t>(slot) * kraw_layer_elems,
                /*group_mean=*/0u);
            !st.ok) {
            return scorer_backend_unavailable(
                failure_reason, "exactmass_scorer_kernel_failed", st);
        }
    }
    std::vector<float> score(nb, 0.0f);
    if (auto st = backend_.copy_to_host(*g_score_dev_, score.data(), 0, nb);
        !st.ok) {
        return scorer_backend_unavailable(
            failure_reason, "exactmass_score_d2h_failed", st);
    }
    // The kernel already normalized (softmax mass averaged over L·M·n_heads), so the
    // scores ARE the block distribution: no inv_lm, no extra normalization tail.
    std::vector<double> best(block_store_->block_count(), 0.0);
    for (uint32_t id = 0; id < nb && id < best.size(); ++id) {
        best[id] = static_cast<double>(score[id]);
    }
    block_store_->set_retrieval_scores(best);
    if (std::getenv("QW3_KVMEM_TRACE")) {
        std::fprintf(stderr,
                     "[bs-exactmass] query-conditioned raw-key ExactMass (per-head): "
                     "L=%u M=%u eff_heads=%u tokens scored %u blocks\n",
                     L, M, eff_heads, nb);
    }
    return true;
}

void QwenExecutor::restore_state_checkpoint(const StateCheckpointSet &checkpoints,
                                            uint32_t index) {
    if (!checkpoints.ready || index >= checkpoints.count) {
        throw std::runtime_error("cannot restore an empty QwenExecutor checkpoint");
    }
    ensure_scratch();
    if (h_ && checkpoints.h_stride > 0) {
        const DeviceTensor *hidden_source = checkpoints.h_shared
            ? checkpoints.h_shared.get()
            : checkpoints.h
            ? checkpoints.h.get()
            : h_batch_.get();
        if (!hidden_source) {
            throw std::runtime_error(
                "cannot restore checkpoint without hidden state");
        }
        const uint64_t hidden_index =
            (checkpoints.h_shared
                 ? static_cast<uint64_t>(checkpoints.h_checkpoint_row)
                 : 0) +
            index;
        require_status(backend_.copy_d2d(
            *h_, *hidden_source,
            hidden_index * checkpoints.h_stride,
            h_->count));
    }
    for (size_t i = 0; i < recurrent_states_.size(); ++i) {
        if (recurrent_states_[i] &&
            i < checkpoints.recurrent_states_shared.size() &&
            checkpoints.recurrent_states_shared[i]) {
            const uint64_t state_count = recurrent_states_[i]->count;
            const uint32_t row_stride =
                checkpoints.checkpoint_stride > 0
                    ? checkpoints.checkpoint_stride
                    : checkpoints.count;
            const uint64_t src_offset =
                (static_cast<uint64_t>(checkpoints.checkpoint_row) *
                     row_stride +
                 index) *
                state_count;
            require_status(backend_.copy_d2d(
                *recurrent_states_[i],
                *checkpoints.recurrent_states_shared[i], src_offset,
                state_count));
        } else if (recurrent_states_[i] &&
                   i < checkpoints.recurrent_states.size() &&
                   checkpoints.recurrent_states[i]) {
            const uint64_t state_count = recurrent_states_[i]->count;
            require_status(backend_.copy_d2d(*recurrent_states_[i],
                                             *checkpoints.recurrent_states[i],
                                             static_cast<uint64_t>(index) * state_count,
                                             state_count));
        }
        if (conv_states_[i] &&
            i < checkpoints.conv_states_shared.size() &&
            checkpoints.conv_states_shared[i]) {
            const uint64_t state_count = conv_states_[i]->count;
            const uint32_t row_stride =
                checkpoints.checkpoint_stride > 0
                    ? checkpoints.checkpoint_stride
                    : checkpoints.count;
            const uint64_t src_offset =
                (static_cast<uint64_t>(checkpoints.checkpoint_row) *
                     row_stride +
                 index) *
                state_count;
            require_status(backend_.copy_d2d(
                *conv_states_[i], *checkpoints.conv_states_shared[i],
                src_offset, state_count));
        } else if (conv_states_[i] && i < checkpoints.conv_states.size() &&
                   checkpoints.conv_states[i]) {
            const uint64_t state_count = conv_states_[i]->count;
            require_status(backend_.copy_d2d(*conv_states_[i],
                                             *checkpoints.conv_states[i],
                                             static_cast<uint64_t>(index) * state_count,
                                             state_count));
        }
    }
    position_ = checkpoints.base_position + index + 1;
    // kvmem block-store registration tracks how many committed tokens have been
    // folded into the block store, which only ever grows via register_append.
    // The MTP verify batch is mtp_single_chunk, so it registered NOTHING; the
    // store still holds exactly `base_position` tokens. Rolling this counter to
    // the post-restore `position_` would make the executor believe the
    // accepted+1 newly-committed tokens are already registered, so the
    // subsequent kvmem_mtp_advance_to -> register_append guard
    // (position_ <= kvmem_registered_pos_) would no-op and those tokens would
    // never enter the store. The next kvmem_reselect would then rebuild the
    // window from an under-counted total_tokens_, leaving window_query_pos_
    // behind position_ (RoPE-frame desync, growing each reject). Restore to the
    // store's true registered length so advance_to registers the gap. Mirrors
    // restore_state, which restores the pre-verify kvmem_registered_pos.
    if (kvmem_enabled_) kvmem_registered_pos_ = checkpoints.base_position;
    const uint32_t logical_pages =
        (position_ + kv_pages_.page_size - 1) / kv_pages_.page_size;
    kv_pages_.truncate_to_logical_pages(logical_pages);
    // Roll the kvmem window tail back to the accepted row, mirroring the
    // position_/kv_pages_ rollback above. The verify batch only GREW the window
    // at the tail (kvmem_extend_window_for_decode_n, no mid-batch re-RoPE), so
    // restoring window_query_pos_ to base+(index+1) and trimming the host page
    // list + count to cover exactly those slots reverts the batch's extension.
    // The surviving slots keep their original physical-page aliasing and their
    // device entries are untouched, so no re-upload is needed (the dropped tail
    // pages are simply no longer addressed — window_page_count_ caps reads).
    if (checkpoints.kvmem_active) {
        const uint32_t page_size = kv_pages_.page_size;
        window_query_pos_ = checkpoints.window_base_query_pos + index + 1;
        window_page_count_ = (window_query_pos_ + page_size - 1) / page_size;
        if (window_pages_host_.size() > window_page_count_) {
            window_pages_host_.resize(window_page_count_);
        }
        // MTP draft window is lockstep with the main window; roll its tail back.
        // Clamp the count to the actual host length: the MTP window only grows
        // during the draft (not this verify), so its host can be shorter than
        // window_page_count_; the next draft's extend regrows it. Keeping
        // count == host size is the invariant the extend relies on.
        mtp_window_page_count_ = std::min<uint32_t>(
            window_page_count_,
            static_cast<uint32_t>(mtp_window_pages_host_.size()));
        if (mtp_window_pages_host_.size() > mtp_window_page_count_) {
            mtp_window_pages_host_.resize(mtp_window_page_count_);
        }
    }
    mtp_prefix_len_ = std::min<uint32_t>(mtp_prefix_len_, position_);
}

NativeExecutorReport QwenExecutor::prime_mtp_prefix_from_current(uint32_t token,
                                                                 uint32_t base_position) {
    return prime_mtp_prefix_from_current_at(
        token, base_position, base_position);
}

NativeExecutorReport QwenExecutor::prime_mtp_prefix_from_current_at(
        uint32_t token,
        uint32_t logical_position,
        uint32_t rope_position) {
    NativeExecutorReport report;
    ensure_mtp_scratch();
    if (logical_position >= kv_ctx_size_) {
        report.missing_kernels.push_back("native MTP prefix KV cache is too small");
        return report;
    }
    const QwenConfig &cfg = model_.config();
    if (cfg.n_ctx_train > 0 && rope_position >= cfg.n_ctx_train) {
        report.missing_kernels.push_back(
            "native MTP prefix RoPE position exceeds the model context limit");
        return report;
    }
    if (logical_position > 0 && logical_position > mtp_prefix_len_) {
        report.missing_kernels.push_back("native MTP prefix chunks are not contiguous");
        return report;
    }

    const DeviceTensor &h_input =
        (logical_position == 0) ? *mtp_zero_h_ : *mtp_prefix_h_;
    NativeExecutorReport step = forward_mtp_draft_from(token, h_input,
                                                       rope_position,
                                                       logical_position,
                                                       rope_position + 1,
                                                       /*compute_logits=*/false,
                                                       /*argmax_out=*/nullptr,
                                                       /*argmax_out_index=*/0,
                                                       /*token_source=*/nullptr,
                                                       /*token_source_index=*/0,
                                                       /*window_frame=*/false,
                                                       /*kv_only=*/true);
    report.ops_executed += step.ops_executed;
    if (!step.ok) {
        report.missing_kernels = std::move(step.missing_kernels);
        return report;
    }
    require_status(backend_.copy_d2d(*mtp_prefix_h_, *h_, 0, h_->count));
    mtp_prefix_len_ =
        std::max<uint32_t>(mtp_prefix_len_, logical_position + 1);
    report.ok = true;
    return report;
}

NativeExecutorReport QwenExecutor::replay_tokens_with_mtp_prefix(
        const std::vector<uint32_t> &tokens,
        uint32_t base_position,
        bool rebuild_prefix,
        double *prefix_seconds,
        uint64_t *prefix_ops) {
    NativeExecutorReport report;
    if (tokens.empty()) {
        report.ok = true;
        return report;
    }

    require_status(backend_.begin());
    bool replay_ok = true;
    try {
        for (uint32_t i = 0; i < tokens.size(); ++i) {
            NativeExecutorReport target =
                forward_one_token(tokens[i], /*compute_logits=*/false);
            report.ops_executed += target.ops_executed;
            if (!target.ok) {
                replay_ok = false;
                report.ok = false;
                report.missing_kernels = std::move(target.missing_kernels);
                break;
            }

            if (rebuild_prefix) {
                const double t0 = steady_seconds();
                const uint32_t logical_pos = base_position + i;
                NativeExecutorReport prefix = kvmem_mtp_local_positions_
                    ? prime_mtp_prefix_from_current_at(
                          tokens[i], logical_pos,
                          last_committed_rope_position())
                    : prime_mtp_prefix_from_current(tokens[i], logical_pos);
                if (prefix_seconds) *prefix_seconds += steady_seconds() - t0;
                if (prefix_ops) *prefix_ops += prefix.ops_executed;
                if (!prefix.ok) {
                    replay_ok = false;
                    report.ok = false;
                    report.missing_kernels = std::move(prefix.missing_kernels);
                    break;
                }
            }
        }
        require_status(backend_.end());
    } catch (...) {
        backend_.end();
        throw;
    }

    if (replay_ok && report.missing_kernels.empty()) report.ok = true;
    return report;
}

NativeExecutorReport QwenExecutor::prime_mtp_prefix_from_last_batch(const std::vector<uint32_t> &tokens,
                                                                    uint32_t base_position,
                                                                    uint32_t batch_min_override) {
    return prime_mtp_prefix_from_last_batch_at(
        tokens, base_position, base_position, batch_min_override);
}

NativeExecutorReport QwenExecutor::prime_mtp_prefix_from_last_batch_at(
        const std::vector<uint32_t> &tokens,
        uint32_t logical_base_position,
        uint32_t rope_base_position,
        uint32_t batch_min_override) {
    NativeExecutorReport report;
    if (tokens.empty()) {
        report.ok = true;
        return report;
    }
    ensure_mtp_scratch();
    if (!h_batch_ || batch_capacity_ == 0 || tokens.size() > batch_capacity_) {
        report.missing_kernels.push_back("native MTP prefix requires the last target batch hidden states");
        return report;
    }
    if (static_cast<uint64_t>(logical_base_position) + tokens.size() >
        kv_ctx_size_) {
        report.missing_kernels.push_back("native MTP prefix KV cache is too small");
        return report;
    }
    if (logical_base_position > 0 &&
        logical_base_position > mtp_prefix_len_) {
        report.missing_kernels.push_back("native MTP prefix chunks are not contiguous");
        return report;
    }
    const QwenConfig &cfg = model_.config();
    if (cfg.n_ctx_train > 0 &&
        static_cast<uint64_t>(rope_base_position) + tokens.size() >
            cfg.n_ctx_train) {
        report.missing_kernels.push_back(
            "native MTP prefix RoPE positions exceed the model context limit");
        return report;
    }

    const uint32_t batch = static_cast<uint32_t>(tokens.size());
    const uint32_t h_stride = static_cast<uint32_t>(h_batch_->count / batch_capacity_);
    auto prime_sequential = [&]() -> NativeExecutorReport {
        NativeExecutorReport seq_report;
        for (uint32_t i = 0; i < batch; ++i) {
            const uint32_t logical_pos = logical_base_position + i;
            const uint32_t rope_pos = rope_base_position + i;
            const DeviceTensor *h_input = mtp_zero_h_.get();
            if (logical_pos > 0) {
                if (i == 0) {
                    h_input = mtp_prefix_h_.get();
                } else {
                    require_status(backend_.copy_d2d(*mtp_prefix_h_, *h_batch_,
                                                     static_cast<uint64_t>(i - 1) * h_stride,
                                                     h_stride));
                    h_input = mtp_prefix_h_.get();
                }
            }

            NativeExecutorReport step = forward_mtp_draft_from(tokens[i], *h_input,
                                                               rope_pos,
                                                               logical_pos,
                                                               rope_pos + 1,
                                                               /*compute_logits=*/false,
                                                               /*argmax_out=*/nullptr,
                                                               /*argmax_out_index=*/0,
                                                               /*token_source=*/nullptr,
                                                               /*token_source_index=*/0,
                                                               /*window_frame=*/false,
                                                               /*kv_only=*/true);
            seq_report.ops_executed += step.ops_executed;
            if (!step.ok) {
                seq_report.missing_kernels = std::move(step.missing_kernels);
                return seq_report;
            }
        }
        require_status(backend_.copy_d2d(*mtp_prefix_h_, *h_batch_,
                                         static_cast<uint64_t>(batch - 1) * h_stride,
                                         h_stride));
        mtp_prefix_len_ = std::max<uint32_t>(
            mtp_prefix_len_, logical_base_position + batch);
        seq_report.ok = true;
        return seq_report;
    };

    const uint32_t batch_min =
        batch_min_override > 0 ? batch_min_override : mtp_prefix_batch_min_tokens();
    if (!mtp_prefix_batch_enabled() || batch < batch_min) {
        return prime_sequential();
    }

    const NativePlanInfo &plan = model_.plan();
    if (!plan.mtp_supported) {
        report.missing_kernels.push_back("native MTP plan is incomplete");
        return report;
    }
    const QwenMtpWeights *mtp = weights_.mtp();
    if (!mtp || !mtp->eh_proj || !mtp->embed_tokens || !mtp->enorm ||
        !mtp->hnorm || !mtp->shared_head_head || !mtp->shared_head_norm) {
        report.missing_kernels.push_back("native MTP weights are incomplete");
        return report;
    }
    if (mtp->layer.recurrent) {
        report.missing_kernels.push_back("native MTP recurrent draft layer is not supported");
        return report;
    }
    // Prefix priming runs strictly after the corresponding target-model batch,
    // and its fast path stops immediately after appending MTP K/V.  At this
    // point only h_batch_ remains live; every other target-prefill activation
    // buffer is dead until the next target chunk.  Reuse those buffers instead
    // of keeping a second, capacity-sized MTP scratch set resident for the
    // whole request (about 0.8 GiB at batch=2048 on Qwen3.6-27B).
    if (!norm_batch_ || !attn_out_batch_ || !ffn_out_batch_ ||
        !proj_batch_ || !gate_proj_batch_ || !q_batch_ || !k_batch_ ||
        !v_batch_) {
        report.missing_kernels.push_back(
            "native MTP batch prefix requires target-prefill scratch");
        return report;
    }

    const QwenLayerWeights &layer = mtp->layer;
    const uint32_t standard_head_dim = cfg.head_dim;
    const uint32_t standard_n_heads = cfg.n_heads;
    const uint32_t standard_n_kv_heads = cfg.n_kv_heads;
    const float eps = cfg.rms_eps;

    auto target_row_stride = [this](const DeviceTensor *t) -> uint32_t {
        return static_cast<uint32_t>(t->count / batch_capacity_);
    };
    const uint32_t mtp_h_stride = target_row_stride(attn_out_batch_.get());
    const uint32_t concat_stride = target_row_stride(proj_batch_.get());
    const uint32_t q_stride_buf = target_row_stride(q_batch_.get());
    const uint32_t k_stride_buf = target_row_stride(k_batch_.get());
    const uint32_t v_stride_buf = target_row_stride(v_batch_.get());

    DeviceTensor &h_inputs = *norm_batch_;
    DeviceTensor &mtp_h = *attn_out_batch_;
    DeviceTensor &mtp_norm = *ffn_out_batch_;
    DeviceTensor &mtp_concat = *proj_batch_;
    DeviceTensor &mtp_enorm = *gate_proj_batch_;
    DeviceTensor &mtp_q = *q_batch_;
    DeviceTensor &mtp_k_batch = *k_batch_;
    DeviceTensor &mtp_v_batch = *v_batch_;

    require_status(backend_.begin());
    begin_record_timing(full_executor_trace_enabled());

    const DeviceTensor &first_h =
        (logical_base_position == 0) ? *mtp_zero_h_ : *mtp_prefix_h_;
    require_status(backend_.pack_mtp_prefix_hinputs(h_inputs, first_h, *h_batch_,
                                                    batch, h_stride));
    record(report, "mtp.prefix_hinput_batch");

    std::vector<uint64_t> rows(batch);
    for (uint32_t i = 0; i < batch; ++i) rows[i] = tokens[i];
    require_status(backend_.q8_0_get_rows_batch(mtp_norm, *mtp->embed_tokens,
                                                rows.data(), batch));
    record(report, "mtp.token_embedding_lookup_batch");
    require_status(backend_.rms_norm_batch(mtp_enorm, mtp_norm,
                                           *mtp->enorm, batch, h_stride, eps));
    record(report, "mtp.enorm_batch");
    require_status(backend_.rms_norm_batch(mtp_h, h_inputs,
                                           *mtp->hnorm, batch, h_stride, eps));
    record(report, "mtp.hnorm_batch");

    require_status(backend_.pack_mtp_concat(mtp_concat,
                                            mtp_enorm,
                                            mtp_h,
                                            batch,
                                            h_stride,
                                            mtp_h_stride,
                                            concat_stride,
                                            h_stride));
    record(report, "mtp.concat_batch");
    require_status(backend_.q8_0_matmul(mtp_h, *mtp->eh_proj,
                                        mtp_concat,
                                        batch, concat_stride, mtp_h_stride));
    record(report, "mtp.eh_proj_batch");

    require_status(backend_.rms_norm_batch(mtp_norm, mtp_h,
                                           *layer.attn_norm, batch, mtp_h_stride, eps));
    record(report, "mtp.attn_norm_batch");
    {
        DeviceTensor *outs[3] = {&mtp_q, &mtp_k_batch, &mtp_v_batch};
        const DeviceWeight *ws[3] = {layer.attn_q, layer.attn_k, layer.attn_v};
        const uint32_t strides[3] = {q_stride_buf, k_stride_buf, v_stride_buf};
        require_status(backend_.q8_0_matmul_fanout(outs, ws, strides, 3,
                                                   mtp_norm, batch, mtp_h_stride));
    }
    record(report, "mtp.attention_qkv_projection_batch");
    require_status(backend_.rmsnorm_per_head_batch(mtp_q, *layer.attn_q_norm,
                                                   batch, q_stride_buf,
                                                   standard_n_heads,
                                                   2 * standard_head_dim,
                                                   standard_head_dim, eps));
    require_status(backend_.rmsnorm_per_head_batch(mtp_k_batch, *layer.attn_k_norm,
                                                   batch, k_stride_buf,
                                                   standard_n_kv_heads,
                                                   standard_head_dim,
                                                   standard_head_dim, eps));
    kvmem_capture_raw_mtp_k(
        mtp_k_batch, logical_base_position, batch);
    trace_rope_position_if_out_of_range(
        "prime_mtp_prefix_from_last_batch.qk", rope_base_position, batch,
        cfg.n_ctx_train, static_cast<int32_t>(weights_.n_layers()),
        /*kernel_uses=*/2);
    require_status(backend_.rope_partial_batch(mtp_q,
                                               batch, q_stride_buf,
                                               standard_n_heads,
                                               2 * standard_head_dim,
                                               cfg.rope_dim, rope_base_position,
                                               cfg.rope_theta));
    require_status(backend_.rope_partial_batch(mtp_k_batch,
                                               batch, k_stride_buf,
                                               standard_n_kv_heads,
                                               standard_head_dim,
                                               cfg.rope_dim, rope_base_position,
                                               cfg.rope_theta));
    const uint32_t per_pos = standard_n_kv_heads * standard_head_dim;
    mtp_kv_pages_.ensure_pages(
        backend_, kv_ctx_size_, logical_base_position, batch);
    if (external_mtp_kv_cache_) {
        mtp_kv_pages_.validate_physical_capacity(
            external_mtp_kv_cache_->physical_slots, "external MTP");
    }
    // Tiered MTP uses a bounded paged pool; a dense append by absolute position
    // would overrun it. Force paged append (see the single-token path).
    const bool use_paged_prefix =
        mtp_paged_prefix_enabled() || kvmem_mtp_tiered_;
    DeviceTensor &mtp_k = mtp_k_cache();
    DeviceTensor &mtp_v = mtp_v_cache();
    if (use_paged_prefix) {
        require_status(backend_.kv_append_batch_paged_device(
            mtp_k, mtp_k_batch, logical_base_position, per_pos, batch,
            mtp_kv_pages_.device_indices(), mtp_kv_pages_.count(),
            mtp_kv_pages_.page_size));
        require_status(backend_.kv_append_batch_paged_device(
            mtp_v, mtp_v_batch, logical_base_position, per_pos, batch,
            mtp_kv_pages_.device_indices(), mtp_kv_pages_.count(),
            mtp_kv_pages_.page_size));
    } else {
        require_status(backend_.kv_append_batch(mtp_k, mtp_k_batch,
                                                logical_base_position,
                                                per_pos, batch));
        require_status(backend_.kv_append_batch(mtp_v, mtp_v_batch,
                                                logical_base_position,
                                                per_pos, batch));
    }
    record(report, "mtp.kv_append_batch");

    // Prefix-prime fast path (see the kv_only branch in forward_mtp_draft_from):
    // this batched forward exists only to populate the MTP KV cache at the true
    // accepted-token positions. The self-attention + residual + FFN that the
    // legacy code ran here over all prior KV (the O(ctx) cost) feed a hidden
    // state that is never read — mtp_prefix_h_ below is taken from the MAIN
    // hidden (h_batch_), and the next draft chain reads only the K/V appended
    // above. Stopping after kv_append leaves the persistent state (mtp KV,
    // mtp_prefix_h_, mtp_prefix_len_) bit-identical, so acceptance and output are
    // unchanged; only the wasted attention is removed.
    require_status(backend_.end());
    require_status(backend_.copy_d2d(*mtp_prefix_h_, *h_batch_,
                                     static_cast<uint64_t>(batch - 1) * h_stride,
                                     h_stride));
    mtp_prefix_len_ = std::max<uint32_t>(
        mtp_prefix_len_, logical_base_position + batch);
    report.ok = true;
    return report;
}

} // namespace qw3
