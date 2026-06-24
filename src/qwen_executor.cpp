#include "qwen_executor.hpp"
#include "env_flags.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qw3 {
namespace {

void require_status(const DeviceStatus &st) {
    if (!st.ok) throw std::runtime_error(st.message);
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

bool kvmem_tier_trace_enabled() {
    return env_flag_enabled("QW3_KVMEM_TIER_TRACE");
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
    for (auto &s : recurrent_states_) {
        if (s) (void) backend_.zero_tensor(*s);
    }
    for (auto &s : conv_states_) {
        if (s) (void) backend_.zero_tensor(*s);
    }
    // KV caches stay allocated; just reset the position so the next forward
    // overwrites slot 0 (the seq_len passed to attention_decode is position+1).
    position_ = 0;
    kv_pages_.reset();
    mtp_kv_pages_.reset();
    mtp_prefix_len_ = 0;
    decode_graph_warmup_pending_ = true;
    // Block-sparse runtime state is per-session: drop the working set + window
    // table. The block store itself (configured selection params) is kept; a
    // fresh register_append sequence rebuilds the block table for the new run.
    kvmem_active_ = false;
    window_pages_host_.clear();
    window_page_count_ = 0;
    window_query_pos_ = 0;
    if (kvmem_cpu_tier_) kvmem_cpu_tier_->clear();
    if (kvmem_nvme_tier_) kvmem_nvme_tier_->clear();
    kvmem_stage_buffer_.clear();
    kvmem_prefetch_ = KvMemPrefetchState{};
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
    if (block_store_) {
        *block_store_ = KvMemStore(block_store_->config());
    }
    kvmem_attn_trace_seen_tokens_ = 0;
    kvmem_attn_trace_sample_ = 0;
    global_attn_trace_seen_tokens_ = 0;
    global_attn_trace_sample_ = 0;
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
        if (page < 0 || static_cast<uint64_t>(page) >= physical_pages) {
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

    require_status(backend_.begin());
    begin_record_timing(executor_trace_timing_enabled());

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

            // Partial RoPE on the first rope_dim of each head's first segment.
            // Baked at the window position when block-sparse is active.
            require_status(backend_.rope_partial(*q_, standard_n_heads,
                                                 2 * standard_head_dim,
                                                 cfg.rope_dim, attn_pos, cfg.rope_theta));
            require_status(backend_.rope_partial(*k_, standard_n_kv_heads,
                                                 standard_head_dim,
                                                 cfg.rope_dim, attn_pos, cfg.rope_theta));

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
                k_cache(il), v_cache(il),
                pages_dev, pages_count, kv_page_size(),
                standard_n_heads, standard_n_kv_heads, standard_head_dim,
                attn_pos, 1,
                standard_n_heads * 2 * standard_head_dim,
                standard_n_heads * standard_head_dim, scale));
            record(report, "layer." + std::to_string(il) + ".attention_sdpa_paged");

            if (global_trace_this_token) {
                global_trace_attention_layer(
                    il, k_cache(il), *q_, 2 * standard_head_dim,
                    pages_dev, pages_count, attn_pos + 1, scale);
            }

            if (bs && kvmem_trace_this_token) {
                kvmem_trace_attention_layer(
                    il, k_cache(il), *q_, 2 * standard_head_dim,
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

NativeExecutorReport QwenExecutor::forward_n_tokens(const std::vector<uint32_t> &tokens,
                                                    bool compute_logits,
                                                    std::vector<DeviceArgmax> *row_argmaxes,
                                                    StateCheckpointSet *state_checkpoints,
                                                    uint32_t state_checkpoint_count,
                                                    bool copy_last_logits) {
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
    const bool bs = kvmem_active_;

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
    // throughput tax of chunking itself).
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
    if (mtp_single_chunk) chunk_size = total;  // MTP verify: never split
    if (chunk_size > total) chunk_size = total;
    if (chunk_size == 0) chunk_size = total;
    ensure_batch_scratch(chunk_size);

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
        state_checkpoints->kvmem_active = bs;
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
        if (bs) kvmem_extend_window_for_decode_n(batch);
        const uint32_t attn_base_pos = bs ? window_query_pos_ : base_pos;
        const DeviceTensor &attn_pages_dev =
            bs ? *window_pages_device_ : kv_page_indices_device();
        const uint32_t attn_pages_count = bs ? window_page_count_ : kv_page_count();

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
                                                     save_state_checkpoints));
            if (record_ops) record(report, "layer." + std::to_string(il) + ".deltanet_batch");
            require_status(backend_.q8_0_matmul(*attn_out_batch_, *layer.ssm_out, *core_batch_,
                                                 batch, core_stride, h_stride));
            if (record_ops) record(report, "layer." + std::to_string(il) + ".recurrent_output_batch");
        } else {
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

            require_status(backend_.rope_partial_batch(*q_batch_,
                                                        batch, q_stride_buf,
                                                        standard_n_heads,
                                                        2 * standard_head_dim,
                                                        cfg.rope_dim, attn_base_pos, cfg.rope_theta));
            require_status(backend_.rope_partial_batch(*k_batch_,
                                                        batch, k_stride_buf,
                                                        standard_n_kv_heads,
                                                        standard_head_dim,
                                                        cfg.rope_dim, attn_base_pos, cfg.rope_theta));

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
                        k_cache(il), v_cache(il),
                        attn_pages_dev, attn_pages_count, kv_page_size(),
                        standard_n_heads, standard_n_kv_heads,
                        standard_head_dim, attn_base_pos, batch,
                        q_stride_buf, mid_stride, scale));
                } else {
                    DeviceStatus attn_st = backend_.attention_prefill_batch_paged_gated_device(
                        *mid_batch_, *q_batch_, 2 * standard_head_dim,
                        k_cache(il), v_cache(il),
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
                            k_cache(il), v_cache(il),
                            bs ? window_pages_host_.data() : kv_page_indices(),
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
                    k_cache(il), v_cache(il),
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
        if (bs) window_query_pos_ += total;
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
        require_status(backend_.end());

        position_ += total;
        if (bs) window_query_pos_ += total;
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
    if (bs) window_query_pos_ += total;
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
    uint32_t current = token_id;
    for (uint32_t i = 0; i < max_tokens; ++i) {
        const uint32_t cache_pos = position_ + i;
        if (cache_pos >= kv_ctx_size_) break;
        const DeviceTensor &h_input = (i == 0) ? *h_ : *mtp_h_;
        NativeExecutorReport report = forward_mtp_draft_from(current, h_input,
                                                             cache_pos, cache_pos,
                                                             cache_pos + 1);
        const bool ok = report.ok;
        const int next = report.argmax_token;
        reports.push_back(std::move(report));
        if (!ok || next < 0) break;
        if (i == 0) {
            mtp_prefix_len_ = std::max<uint32_t>(mtp_prefix_len_, position_ + 1);
        }
        current = static_cast<uint32_t>(next);
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

    std::vector<NativeExecutorReport> reports;
    reports.reserve(max_tokens);
    for (uint32_t i = 0; i < max_tokens; ++i) {
        const uint32_t cache_pos = position_ + i;
        if (cache_pos >= kv_ctx_size_) break;
        const DeviceTensor &h_input = (i == 0) ? *h_ : *mtp_h_;
        const DeviceArgmaxBuffer *token_source = i == 0 ? nullptr : mtp_draft_argmaxes_.get();
        NativeExecutorReport report = forward_mtp_draft_from(token_id,
                                                             h_input,
                                                             cache_pos,
                                                             cache_pos,
                                                             cache_pos + 1,
                                                             /*compute_logits=*/true,
                                                             mtp_draft_argmaxes_.get(),
                                                             i,
                                                             token_source,
                                                             i == 0 ? 0 : i - 1);
        const bool ok = report.ok;
        reports.push_back(std::move(report));
        if (!ok) break;
        if (i == 0) {
            mtp_prefix_len_ = std::max<uint32_t>(mtp_prefix_len_, position_ + 1);
        }
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
                                                          uint32_t token_source_index) {
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
    require_status(backend_.rope_partial(*q_, standard_n_heads,
                                         2 * standard_head_dim,
                                         cfg.rope_dim, rope_pos, cfg.rope_theta));
    require_status(backend_.rope_partial(*k_, standard_n_kv_heads,
                                         standard_head_dim,
                                         cfg.rope_dim, rope_pos, cfg.rope_theta));

    const uint32_t per_pos = standard_n_kv_heads * standard_head_dim;
    mtp_kv_pages_.ensure_pages(backend_, kv_ctx_size_, cache_pos, 1);
    if (external_mtp_kv_cache_) {
        mtp_kv_pages_.validate_physical_capacity(
            external_mtp_kv_cache_->physical_slots, "external MTP");
    }
    const bool use_paged_prefix = mtp_paged_prefix_enabled();
    DeviceTensor &mtp_k = mtp_k_cache();
    DeviceTensor &mtp_v = mtp_v_cache();
    if (use_paged_prefix) {
        require_status(backend_.kv_append_batch_paged_device(
            mtp_k, *k_, cache_pos, per_pos,
            1, mtp_kv_pages_.device_indices(),
            mtp_kv_pages_.count(), mtp_kv_pages_.page_size));
        require_status(backend_.kv_append_batch_paged_device(
            mtp_v, *v_, cache_pos, per_pos,
            1, mtp_kv_pages_.device_indices(),
            mtp_kv_pages_.count(), mtp_kv_pages_.page_size));
    } else {
        require_status(backend_.kv_append(mtp_k, *k_, cache_pos, per_pos));
        require_status(backend_.kv_append(mtp_v, *v_, cache_pos, per_pos));
    }
    record(report, "mtp.kv_append");

    const float scale = 1.0f / std::sqrt(static_cast<float>(standard_head_dim));
    if (use_paged_prefix) {
        require_status(backend_.attention_decode_batch_paged_gated_device(
            *mid_, *q_, 2 * standard_head_dim, mtp_k,
            mtp_v, mtp_kv_pages_.device_indices(),
            mtp_kv_pages_.count(), mtp_kv_pages_.page_size,
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
    effective.estimated_block_bytes =
        estimate_kvmem_block_bytes(model_cfg, standard_layers,
                                   effective.block_tokens);
    const uint64_t total_device_bytes = backend_.total_device_bytes();
    if (effective.gpu_memory_ratio > 0.0 &&
        total_device_bytes > 0 &&
        effective.estimated_block_bytes > 0) {
        const uint64_t cap_bytes = static_cast<uint64_t>(
            static_cast<long double>(total_device_bytes) *
            effective.gpu_memory_ratio);
        const uint64_t cap_blocks64 = cap_bytes / effective.estimated_block_bytes;
        const uint32_t cap_blocks = cap_blocks64 >
                static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
            ? std::numeric_limits<uint32_t>::max()
            : static_cast<uint32_t>(cap_blocks64);
        effective.estimated_gpu_block_capacity = cap_blocks;
        if (cap_blocks > 0) {
            const uint64_t cap_tokens =
                static_cast<uint64_t>(cap_blocks) * effective.block_tokens;
            if (cap_tokens < effective.select_budget) {
                effective.select_budget = static_cast<uint32_t>(std::max<uint64_t>(
                    effective.block_tokens, cap_tokens));
            }
        }
    }

    block_store_ = std::make_unique<KvMemStore>(effective);
    kvmem_active_ = false;
    window_pages_host_.clear();
    window_page_count_ = 0;
    window_query_pos_ = 0;
    kvmem_cpu_tier_.reset();
    kvmem_nvme_tier_.reset();
    kvmem_cpu_bytes_.reset();
    kvmem_stage_buffer_.clear();
    kvmem_prefetch_ = KvMemPrefetchState{};
    if (effective.cpu_tier_bytes > 0 && effective.estimated_block_bytes > 0) {
        PinnedKvTierConfig pcfg;
        pcfg.total_bytes = effective.cpu_tier_bytes;
        pcfg.slot_bytes = effective.estimated_block_bytes;
        kvmem_cpu_tier_ = std::make_unique<PinnedKvTier>(pcfg);
        if (kvmem_cpu_tier_->enabled()) {
            kvmem_cpu_bytes_ = backend_.host_buffer(
                static_cast<uint64_t>(kvmem_cpu_tier_->slot_count()) *
                    pcfg.slot_bytes,
                "kvmem_cpu_tier");
        }
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
        kvmem_nvme_tier_ = std::make_unique<NvmeKvTier>(std::move(ncfg));
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
    block_store_->register_append(n_new_tokens);
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
    if (sample->elem_size == 1) {
        throw std::runtime_error(
            "KVMem CPU offload does not yet support 1-byte KV dtypes");
    }
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

uint64_t QwenExecutor::kvmem_block_spill_bytes(const KvMemBlock &block) const {
    if (block.n_tokens == 0) return 0;
    const QwenConfig &cfg = model_.config();
    const uint64_t page_bytes = kvmem_kv_page_bytes();
    const uint32_t page_size = kv_pages_.page_size;
    const uint32_t first_page = block.orig_pos_start / page_size;
    const uint32_t last_page =
        (block.orig_pos_start + block.n_tokens - 1) / page_size;
    const uint32_t n_pages = last_page - first_page + 1;
    return page_bytes * n_pages *
           count_standard_attention_layers(cfg, weights_.n_layers()) * 2ull;
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
    if (block.baked_pos == canonical) return;
    const int64_t from = block.baked_pos;

    const QwenConfig &cfg = model_.config();
    const uint32_t per_pos =
        static_cast<uint32_t>(cfg.n_kv_heads) * cfg.head_dim;
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
    block_store_->set_block_baked_pos(block_id, canonical);
    if (kvmem_tier_trace_enabled()) {
        std::fprintf(stderr,
                     "[kvmem-tier] canonicalize block=%u from=%lld to=%u\n",
                     block_id, static_cast<long long>(from), block.orig_pos_start);
    }
}

void QwenExecutor::kvmem_copy_block_to_host(const KvMemBlock &block,
                                            std::vector<uint8_t> &dst) {
    if (block.n_tokens == 0) {
        dst.clear();
        return;
    }
    const QwenConfig &cfg = model_.config();
    const uint64_t page_bytes = kvmem_kv_page_bytes();
    const uint32_t page_size = kv_pages_.page_size;
    const uint32_t first_page = block.orig_pos_start / page_size;
    const uint32_t last_page = (block.orig_pos_start + block.n_tokens - 1) / page_size;
    const uint64_t per_pos =
        static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
    dst.clear();
    dst.resize(kvmem_block_spill_bytes(block));
    uint8_t *out = dst.data();
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
            require_status(backend_.copy_bytes_to_host_async(kc, out, byte_offset,
                                                             page_bytes));
            out += page_bytes;
            require_status(backend_.copy_bytes_to_host_async(vc, out, byte_offset,
                                                             page_bytes));
            out += page_bytes;
        }
    }
}

void QwenExecutor::kvmem_copy_block_from_host(
        const KvMemBlock &block, const std::vector<uint8_t> &src) {
    kvmem_copy_block_from_host(block, src.data(), src.size());
}

void QwenExecutor::kvmem_copy_block_from_host(
        const KvMemBlock &block, const void *src, uint64_t src_bytes) {
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
            require_status(backend_.copy_bytes_from_host_async(kc, byte_offset, in,
                                                               page_bytes));
            in += page_bytes;
            require_status(backend_.copy_bytes_from_host_async(vc, byte_offset, in,
                                                               page_bytes));
            in += page_bytes;
        }
    }
}

void QwenExecutor::kvmem_stage_in(const KvMemPlan &plan) {
    kvmem_start_prefetch(plan);
    kvmem_finish_prefetch();
}

void QwenExecutor::kvmem_start_prefetch(const KvMemPlan &plan) {
    if (!block_store_) return;
    if (kvmem_prefetch_.active) {
        throw std::runtime_error("KVMem prefetch already active");
    }
    kvmem_prefetch_ = KvMemPrefetchState{};
    kvmem_prefetch_.active = true;
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
                if (!kvmem_cpu_data()) {
                    throw std::runtime_error("KVMem CPU stage-in has no CPU backing buffer");
                }
                const int32_t slot = kvmem_cpu_tier_->block_slot(rm.block_id);
                if (slot < 0) {
                    throw std::runtime_error("KVMem CPU stage-in missing CPU slot");
                }
                const uint64_t offset = kvmem_cpu_tier_->slot_offset(slot);
                if (offset + bytes > kvmem_cpu_bytes()) {
                    throw std::runtime_error("KVMem CPU tier slot read out of range");
                }
                stage_src = kvmem_cpu_data() + offset;
            } else if (block.tier == KvTier::SSD) {
                if (!kvmem_nvme_tier_) {
                    throw std::runtime_error("KVMem NVMe stage-in has no NVMe tier");
                }
                kvmem_prefetch_.nvme_buffers.emplace_back();
                std::vector<uint8_t> &buf = kvmem_prefetch_.nvme_buffers.back();
                buf.resize(bytes);
                kvmem_nvme_tier_->read_block(rm.block_id, buf.data(), bytes);
                stage_src = buf.data();
            } else {
                throw std::runtime_error(
                    "KVMem stage-in requested a non-resident block with no backing tier");
            }
            const uint32_t page_size = kv_pages_.page_size;
            const uint32_t first_page = block.orig_pos_start / page_size;
            const uint32_t last_page =
                (block.orig_pos_start + block.n_tokens - 1) / page_size;
            for (uint32_t lp = first_page; lp <= last_page; ++lp) {
                (void)kv_pages_.ensure_logical_page_resident(backend_, lp);
            }
            kvmem_copy_block_from_host(block, stage_src, stage_bytes);
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
    } catch (...) {
        kvmem_prefetch_ = KvMemPrefetchState{};
        throw;
    }
}

void QwenExecutor::kvmem_finish_prefetch() {
    if (!kvmem_prefetch_.active) return;
    if (kvmem_prefetch_.queued_h2d) {
        require_status(backend_.wait_kv_transfer());
    }
    for (const KvMemPrefetchBlock &pb : kvmem_prefetch_.blocks) {
        if (pb.from == KvTier::CPU && kvmem_cpu_tier_) {
            kvmem_cpu_tier_->release_block(pb.block_id);
        } else if (pb.from == KvTier::SSD && kvmem_nvme_tier_) {
            kvmem_nvme_tier_->release_block(pb.block_id);
        }
        block_store_->set_block_tier(pb.block_id, KvTier::GPU);
    }
    kvmem_prefetch_ = KvMemPrefetchState{};
}

void QwenExecutor::kvmem_stage_out(const std::vector<uint32_t> &block_ids) {
    if (!block_store_ || block_ids.empty()) return;
    if (!kvmem_cpu_tier_ && !kvmem_nvme_tier_) return;
    const auto &blocks = block_store_->blocks();
    for (uint32_t block_id : block_ids) {
        if (block_id >= blocks.size()) continue;
        const KvMemBlock &block = blocks[block_id];
        if (!kvmem_block_pages_resident(block)) continue;
        kvmem_canonicalize_block_for_tier(block_id);
        require_status(backend_.begin_kv_transfer_from_device());
        kvmem_copy_block_to_host(block, kvmem_stage_buffer_);
        require_status(backend_.wait_kv_transfer());
        bool placed = false;
        int32_t cpu_slot = -1;
        int32_t nvme_slot = -1;
        if (kvmem_cpu_tier_) {
            if (!kvmem_cpu_data()) {
                throw std::runtime_error("KVMem CPU stage-out has no CPU backing buffer");
            }
            PinnedSlotPlacement placement;
            if (kvmem_nvme_tier_) {
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
                    const uint64_t victim_offset =
                        kvmem_cpu_tier_->slot_offset(placement.slot);
                    kvmem_nvme_tier_->write_block(
                        victim, kvmem_cpu_data() + victim_offset,
                        victim_bytes);
                    if (kvmem_tier_trace_enabled()) {
                        std::fprintf(stderr,
                                     "[kvmem-tier] cpu_evict block=%u to=nvme slot=%d bytes=%llu\n",
                                     victim, kvmem_nvme_tier_->block_slot(victim),
                                     static_cast<unsigned long long>(victim_bytes));
                    }
                    block_store_->set_block_tier(
                        victim, KvTier::SSD, -1,
                        kvmem_nvme_tier_->block_slot(victim));
                }
                const uint64_t offset = kvmem_cpu_tier_->slot_offset(placement.slot);
                if (offset + kvmem_stage_buffer_.size() > kvmem_cpu_bytes()) {
                    throw std::runtime_error("KVMem CPU tier slot write out of range");
                }
                std::memcpy(kvmem_cpu_data() + offset,
                            kvmem_stage_buffer_.data(),
                            kvmem_stage_buffer_.size());
                placed = true;
                cpu_slot = placement.slot;
            }
        } else if (kvmem_nvme_tier_) {
            kvmem_nvme_tier_->write_block(
                block_id, kvmem_stage_buffer_.data(),
                kvmem_stage_buffer_.size());
            placed = true;
            nvme_slot = kvmem_nvme_tier_->block_slot(block_id);
        }
        if (!placed) continue;
        const uint32_t page_size = kv_pages_.page_size;
        const uint32_t first_page = block.orig_pos_start / page_size;
        const uint32_t last_page =
            (block.orig_pos_start + block.n_tokens - 1) / page_size;
        kv_pages_.release_logical_pages(backend_, first_page,
                                        last_page - first_page + 1);
        block_store_->set_block_tier(
            block_id, cpu_slot >= 0 ? KvTier::CPU : KvTier::SSD,
            cpu_slot, nvme_slot);
        if (kvmem_tier_trace_enabled()) {
            std::fprintf(stderr,
                         "[kvmem-tier] stage_out block=%u to=%s slot=%d bytes=%llu pages=%u\n",
                         block_id, cpu_slot >= 0 ? "cpu" : "nvme",
                         cpu_slot >= 0 ? cpu_slot : nvme_slot,
                         static_cast<unsigned long long>(kvmem_stage_buffer_.size()),
                         last_page - first_page + 1);
        }
    }
}

void QwenExecutor::kvmem_assemble(const KvMemPlan &plan) {
    const QwenConfig &cfg = model_.config();
    const uint32_t page_size = kv_pages_.page_size;
    const uint32_t per_pos =
        static_cast<uint32_t>(cfg.n_kv_heads) * cfg.head_dim;

    // Build the window page-index list = each selected block's ORIGINAL
    // physical pages, in window (ascending block-id) order, packed contiguously
    // from window position 0. No-copy: these alias the same physical pages the
    // block already occupies in kv_pages_; the window is just a pointer
    // reordering. baked_pos in the plan tells us where the block's K currently
    // sits so re-RoPE can de-rotate from there.
    window_pages_host_.clear();
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
        }
        bs_window_block_ids_.push_back(rm.block_id);
        bs_win_base_host_.push_back(rm.to_base);
        bs_blk_tokens_host_.push_back(static_cast<int32_t>(b.n_tokens));
    }
    window_page_count_ = static_cast<uint32_t>(window_pages_host_.size());
    window_query_pos_ = plan.total_window_tokens;
    sync_window_pages_device(window_page_count_);

    // Re-RoPE each moved block in place, per standard attention layer. Skipped
    // blocks (already baked at their window slot) issue no kernel. The window
    // page table addresses the physical pages, so the kernel maps window slot
    // win_base+tok -> page_indices[(win_base+tok)/page_size] just like append.
    for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
        if (!cfg.is_standard_attention_layer(il)) continue;
        DeviceTensor &kc = k_cache(il);
        for (const KvMemRemap &rm : plan.remaps) {
            if (rm.skip) continue;
            require_status(backend_.rope_block_remap_paged_device(
                kc, rm.n_tokens, cfg.n_kv_heads, per_pos, cfg.head_dim,
                cfg.rope_dim, /*win_base=*/static_cast<uint32_t>(rm.to_base),
                /*from_base=*/rm.from_base, /*to_base=*/rm.to_base,
                *window_pages_device_, page_size, cfg.rope_theta));
        }
    }
    kvmem_active_ = true;
    // Rebuild the per-block representative K + reset the score accumulator for
    // the new interval (k̄ is read from the just-re-RoPE'd window K).
    kvmem_recompute_kbar();
}

uint32_t QwenExecutor::kvmem_reselect() {
    if (!kvmem_enabled_ || !block_store_) return 0;
    if (block_store_->block_count() == 0) return 0;
    const KvMemMethod method = block_store_->config().select_method;
    // Build the global content-frame index once, from the pristine post-prefill
    // cache (every block still baked at its true position). After the first
    // assembly re-RoPEs selected blocks into window slots this is no longer
    // possible, but the content mean is position-invariant so one build serves
    // the whole session. Only needed for the Retrieval method.
    if (method == KvMemMethod::Retrieval && !g_content_ready_ &&
        !kvmem_active_) {
        kvmem_build_content_index();
    }
    // Score selection by the configured method (all three feed pick_topk, which
    // always keeps the sink + recent windows):
    //   Retrieval — global content similarity (can resurrect dropped blocks);
    //               falls back to the window-local heat fold when the
    //               index/query isn't live (q8/fp8, or first selection).
    //   H2O       — window-local cumulative attention heat only.
    //   Recency   — no learned signal; pick_topk keeps sink + recent only.
    switch (method) {
        case KvMemMethod::Retrieval:
            // Preserve the just-finished step's window-local profile before
            // the global retrieval scorer overwrites the ranking score. Quota
            // selection can then draw from both pools.
            kvmem_drain_scores();
            (void)kvmem_retrieval_score();
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
    KvMemPlan plan =
        block_store_->set_selection(block_store_->pick_topk_blocks());
    kvmem_start_prefetch(plan);
    kvmem_finish_prefetch();
    kvmem_assemble(plan);
    kvmem_stage_out(plan.stage_out);
    return window_query_pos_;
}

uint32_t QwenExecutor::kvmem_set_selection(
        const std::vector<uint32_t> &block_ids) {
    if (!kvmem_enabled_ || !block_store_) return 0;
    if (block_store_->block_count() == 0) return 0;
    KvMemPlan plan = block_store_->set_selection(block_ids);
    kvmem_start_prefetch(plan);
    kvmem_finish_prefetch();
    kvmem_assemble(plan);
    kvmem_stage_out(plan.stage_out);
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
void QwenExecutor::kvmem_extend_window_for_decode_n(uint32_t n) {
    if (n == 0) return;
    const uint32_t page_size = kv_pages_.page_size;
    const uint32_t need_pages =
        (window_query_pos_ + n + page_size - 1) / page_size;
    if (need_pages <= window_page_count_) return;  // current pages have room
    // Constant page offset between the window tail and the true cache tail;
    // window_query_pos_ <= position_ always (the window is a compressed view).
    const uint32_t delta = (position_ - window_query_pos_) / page_size;
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

    // k̄ needs fp16/fp32 K; q8/fp8 caches can't be averaged meaningfully here,
    // so selection silently stays recency-weighted (the kernel returns an error
    // we tolerate).
    auto st = backend_.block_kmean_paged_device(
        k_cache(static_cast<uint32_t>(bs_score_layer_)), *bs_kbar_,
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
// q8/fp8 caches can't be de-RoPE'd → index stays unbuilt, retrieval falls back to
// the window-local heat signal.
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
    }

    if (n_blocks > g_kbar_global_capacity_) {
        g_kbar_global_capacity_ = n_blocks;
        g_kbar_ = backend_.tensor_f32(
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
}

// Per decode step at the representative layer: de-RoPE the current RoPE-baked Q
// (baked at window_query_pos_) into the content frame so it can be scored against
// the content-frame mean keys at the retrieval boundary. Cheap (one launch per
// step at one layer); inert unless the global index is live.
void QwenExecutor::kvmem_snapshot_content_query(uint32_t layer_index) {
    if (!g_content_ready_) return;
    if (static_cast<int32_t>(layer_index) != bs_score_layer_) return;
    const QwenConfig &cfg = model_.config();
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
bool QwenExecutor::kvmem_retrieval_score() {
    if (!g_content_ready_ || !g_query_ready_ || !block_store_) return false;
    if (g_indexed_blocks_ == 0) return false;
    const QwenConfig &cfg = model_.config();
    // Content-frame Q . k̄ has no 1/sqrt(d) softmax scale to honor (it is a bare
    // similarity rank), so use scale = 1.
    require_status(backend_.zero_tensor(*g_score_dev_));
    if (auto st = backend_.block_attn_score_step_device(
            *g_score_dev_, *g_query_content_, *g_kbar_,
            /*q_stride=*/cfg.head_dim, g_indexed_blocks_,
            cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, /*scale=*/1.0f);
        !st.ok) {
        return false;
    }
    std::vector<float> score(g_indexed_blocks_, 0.0f);
    if (auto st = backend_.copy_to_host(*g_score_dev_, score.data(), 0,
                                        g_indexed_blocks_);
        !st.ok) {
        return false;
    }
    std::vector<double> scores(block_store_->block_count(), 0.0);
    for (uint32_t id = 0; id < g_indexed_blocks_ && id < scores.size(); ++id) {
        scores[id] = static_cast<double>(score[id]);
    }
    if (block_store_->config().retrieval_method ==
        KvMemRetrievalMethod::MeanAttention) {
        double max_score = -std::numeric_limits<double>::infinity();
        for (uint32_t id = 0; id < g_indexed_blocks_ && id < scores.size(); ++id) {
            max_score = std::max(max_score, scores[id]);
        }
        double denom = 0.0;
        if (std::isfinite(max_score)) {
            for (uint32_t id = 0; id < g_indexed_blocks_ && id < scores.size(); ++id) {
                scores[id] = std::exp(scores[id] - max_score);
                denom += scores[id];
            }
        }
        if (denom > 0.0) {
            for (uint32_t id = 0; id < g_indexed_blocks_ && id < scores.size(); ++id) {
                scores[id] /= denom;
            }
        }
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
    }
    mtp_prefix_len_ = std::min<uint32_t>(mtp_prefix_len_, position_);
}

NativeExecutorReport QwenExecutor::prime_mtp_prefix_from_current(uint32_t token,
                                                                 uint32_t base_position) {
    NativeExecutorReport report;
    ensure_mtp_scratch();
    if (base_position >= kv_ctx_size_) {
        report.missing_kernels.push_back("native MTP prefix KV cache is too small");
        return report;
    }
    if (base_position > 0 && base_position > mtp_prefix_len_) {
        report.missing_kernels.push_back("native MTP prefix chunks are not contiguous");
        return report;
    }

    const DeviceTensor &h_input = (base_position == 0) ? *mtp_zero_h_ : *mtp_prefix_h_;
    NativeExecutorReport step = forward_mtp_draft_from(token, h_input,
                                                       base_position,
                                                       base_position,
                                                       base_position + 1,
                                                       /*compute_logits=*/false);
    report.ops_executed += step.ops_executed;
    if (!step.ok) {
        report.missing_kernels = std::move(step.missing_kernels);
        return report;
    }
    require_status(backend_.copy_d2d(*mtp_prefix_h_, *h_, 0, h_->count));
    mtp_prefix_len_ = std::max<uint32_t>(mtp_prefix_len_, base_position + 1);
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
                NativeExecutorReport prefix =
                    prime_mtp_prefix_from_current(tokens[i], base_position + i);
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
    if (base_position + tokens.size() > kv_ctx_size_) {
        report.missing_kernels.push_back("native MTP prefix KV cache is too small");
        return report;
    }
    if (base_position > 0 && base_position > mtp_prefix_len_) {
        report.missing_kernels.push_back("native MTP prefix chunks are not contiguous");
        return report;
    }

    const uint32_t batch = static_cast<uint32_t>(tokens.size());
    const uint32_t h_stride = static_cast<uint32_t>(h_batch_->count / batch_capacity_);
    auto prime_sequential = [&]() -> NativeExecutorReport {
        NativeExecutorReport seq_report;
        for (uint32_t i = 0; i < batch; ++i) {
            const uint32_t pos = base_position + i;
            const DeviceTensor *h_input = mtp_zero_h_.get();
            if (pos > 0) {
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
                                                               pos, pos, pos + 1,
                                                               /*compute_logits=*/false);
            seq_report.ops_executed += step.ops_executed;
            if (!step.ok) {
                seq_report.missing_kernels = std::move(step.missing_kernels);
                return seq_report;
            }
        }
        require_status(backend_.copy_d2d(*mtp_prefix_h_, *h_batch_,
                                         static_cast<uint64_t>(batch - 1) * h_stride,
                                         h_stride));
        mtp_prefix_len_ = std::max<uint32_t>(mtp_prefix_len_, base_position + batch);
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
    ensure_mtp_batch_scratch(batch);
    if (!mtp_h_input_batch_ || !mtp_h_batch_ || !mtp_norm_batch_ ||
        !mtp_concat_batch_ || !mtp_q_batch_ || !mtp_k_batch_ ||
        !mtp_v_batch_ || !mtp_mid_batch_ || !mtp_ffn_gate_batch_ ||
        !mtp_ffn_up_batch_ || !mtp_ffn_mid_batch_ || !mtp_ffn_out_batch_) {
        report.missing_kernels.push_back("native MTP batch prefix scratch allocation failed");
        return report;
    }

    const QwenConfig &cfg = model_.config();
    const QwenLayerWeights &layer = mtp->layer;
    const uint32_t standard_head_dim = cfg.head_dim;
    const uint32_t standard_n_heads = cfg.n_heads;
    const uint32_t standard_n_kv_heads = cfg.n_kv_heads;
    const float eps = cfg.rms_eps;

    auto mtp_row_stride = [this](const DeviceTensor *t) -> uint32_t {
        return static_cast<uint32_t>(t->count / mtp_batch_capacity_);
    };
    const uint32_t mtp_h_stride = mtp_row_stride(mtp_h_batch_.get());
    const uint32_t concat_stride = mtp_row_stride(mtp_concat_batch_.get());
    const uint32_t ffn_stride = mtp_row_stride(mtp_ffn_gate_batch_.get());
    const uint32_t q_stride_buf = mtp_row_stride(mtp_q_batch_.get());
    const uint32_t k_stride_buf = mtp_row_stride(mtp_k_batch_.get());
    const uint32_t v_stride_buf = mtp_row_stride(mtp_v_batch_.get());
    const uint32_t mid_stride = mtp_row_stride(mtp_mid_batch_.get());

    require_status(backend_.begin());
    begin_record_timing(full_executor_trace_enabled());

    DeviceTensor &h_inputs = *mtp_h_input_batch_;
    const DeviceTensor &first_h = (base_position == 0) ? *mtp_zero_h_ : *mtp_prefix_h_;
    require_status(backend_.pack_mtp_prefix_hinputs(h_inputs, first_h, *h_batch_,
                                                    batch, h_stride));
    record(report, "mtp.prefix_hinput_batch");

    std::vector<uint64_t> rows(batch);
    for (uint32_t i = 0; i < batch; ++i) rows[i] = tokens[i];
    require_status(backend_.q8_0_get_rows_batch(*mtp_norm_batch_, *mtp->embed_tokens,
                                                rows.data(), batch));
    record(report, "mtp.token_embedding_lookup_batch");
    require_status(backend_.rms_norm_batch(*mtp_ffn_out_batch_, *mtp_norm_batch_,
                                           *mtp->enorm, batch, h_stride, eps));
    record(report, "mtp.enorm_batch");
    require_status(backend_.rms_norm_batch(*mtp_h_batch_, h_inputs,
                                           *mtp->hnorm, batch, h_stride, eps));
    record(report, "mtp.hnorm_batch");

    require_status(backend_.pack_mtp_concat(*mtp_concat_batch_,
                                            *mtp_ffn_out_batch_,
                                            *mtp_h_batch_,
                                            batch,
                                            h_stride,
                                            mtp_h_stride,
                                            concat_stride,
                                            h_stride));
    record(report, "mtp.concat_batch");
    require_status(backend_.q8_0_matmul(*mtp_h_batch_, *mtp->eh_proj,
                                        *mtp_concat_batch_,
                                        batch, concat_stride, mtp_h_stride));
    record(report, "mtp.eh_proj_batch");

    require_status(backend_.rms_norm_batch(*mtp_norm_batch_, *mtp_h_batch_,
                                           *layer.attn_norm, batch, mtp_h_stride, eps));
    record(report, "mtp.attn_norm_batch");
    {
        DeviceTensor *outs[3] = {mtp_q_batch_.get(), mtp_k_batch_.get(), mtp_v_batch_.get()};
        const DeviceWeight *ws[3] = {layer.attn_q, layer.attn_k, layer.attn_v};
        const uint32_t strides[3] = {q_stride_buf, k_stride_buf, v_stride_buf};
        require_status(backend_.q8_0_matmul_fanout(outs, ws, strides, 3,
                                                   *mtp_norm_batch_, batch, mtp_h_stride));
    }
    record(report, "mtp.attention_qkv_projection_batch");
    require_status(backend_.rmsnorm_per_head_batch(*mtp_q_batch_, *layer.attn_q_norm,
                                                   batch, q_stride_buf,
                                                   standard_n_heads,
                                                   2 * standard_head_dim,
                                                   standard_head_dim, eps));
    require_status(backend_.rmsnorm_per_head_batch(*mtp_k_batch_, *layer.attn_k_norm,
                                                   batch, k_stride_buf,
                                                   standard_n_kv_heads,
                                                   standard_head_dim,
                                                   standard_head_dim, eps));
    require_status(backend_.rope_partial_batch(*mtp_q_batch_,
                                               batch, q_stride_buf,
                                               standard_n_heads,
                                               2 * standard_head_dim,
                                               cfg.rope_dim, base_position, cfg.rope_theta));
    require_status(backend_.rope_partial_batch(*mtp_k_batch_,
                                               batch, k_stride_buf,
                                               standard_n_kv_heads,
                                               standard_head_dim,
                                               cfg.rope_dim, base_position, cfg.rope_theta));
    const uint32_t per_pos = standard_n_kv_heads * standard_head_dim;
    mtp_kv_pages_.ensure_pages(backend_, kv_ctx_size_, base_position, batch);
    if (external_mtp_kv_cache_) {
        mtp_kv_pages_.validate_physical_capacity(
            external_mtp_kv_cache_->physical_slots, "external MTP");
    }
    const bool use_paged_prefix = mtp_paged_prefix_enabled();
    DeviceTensor &mtp_k = mtp_k_cache();
    DeviceTensor &mtp_v = mtp_v_cache();
    if (use_paged_prefix) {
        require_status(backend_.kv_append_batch_paged_device(
            mtp_k, *mtp_k_batch_, base_position, per_pos, batch,
            mtp_kv_pages_.device_indices(), mtp_kv_pages_.count(),
            mtp_kv_pages_.page_size));
        require_status(backend_.kv_append_batch_paged_device(
            mtp_v, *mtp_v_batch_, base_position, per_pos, batch,
            mtp_kv_pages_.device_indices(), mtp_kv_pages_.count(),
            mtp_kv_pages_.page_size));
    } else {
        require_status(backend_.kv_append_batch(mtp_k, *mtp_k_batch_,
                                                base_position, per_pos, batch));
        require_status(backend_.kv_append_batch(mtp_v, *mtp_v_batch_,
                                                base_position, per_pos, batch));
    }
    record(report, "mtp.kv_append_batch");

    const float scale = 1.0f / std::sqrt(static_cast<float>(standard_head_dim));
    if (use_paged_prefix) {
        DeviceStatus attn_st = backend_.attention_prefill_batch_paged_gated_device(
            *mtp_mid_batch_, *mtp_q_batch_, 2 * standard_head_dim,
            mtp_k, mtp_v, mtp_kv_pages_.device_indices(),
            mtp_kv_pages_.count(), mtp_kv_pages_.page_size,
            standard_n_heads, standard_n_kv_heads, standard_head_dim,
            base_position, batch, q_stride_buf, mid_stride, scale);
        if (!attn_st.ok) {
            require_status(backend_.attention_decode_batch_paged_gated_device(
                *mtp_mid_batch_, *mtp_q_batch_, 2 * standard_head_dim,
                mtp_k, mtp_v, mtp_kv_pages_.device_indices(),
                mtp_kv_pages_.count(), mtp_kv_pages_.page_size,
                standard_n_heads, standard_n_kv_heads, standard_head_dim,
                base_position, batch, q_stride_buf, mid_stride, scale));
        }
    } else {
        require_status(backend_.attention_decode_batch_gated(
            *mtp_mid_batch_, *mtp_q_batch_, 2 * standard_head_dim,
            mtp_k, mtp_v, standard_n_heads,
            standard_n_kv_heads, standard_head_dim, base_position, batch,
            q_stride_buf, mid_stride, scale));
    }
    record(report, "mtp.attention_sdpa_batch");
    require_status(backend_.q8_0_matmul_add(*mtp_h_batch_,
                                            *mtp_h_batch_,
                                            *mtp_ffn_out_batch_,
                                            *layer.attn_output,
                                            *mtp_mid_batch_,
                                            batch, mid_stride, mtp_h_stride));
    record(report, "mtp.attn_residual_batch");

    require_status(backend_.rms_norm_batch(*mtp_norm_batch_, *mtp_h_batch_,
                                           *layer.ffn_norm, batch, mtp_h_stride, eps));
    record(report, "mtp.ffn_norm_batch");
    {
        DeviceTensor *outs[2] = {mtp_ffn_gate_batch_.get(), mtp_ffn_up_batch_.get()};
        const DeviceWeight *ws[2] = {layer.ffn_gate, layer.ffn_up};
        const uint32_t strides[2] = {ffn_stride, ffn_stride};
        require_status(backend_.q8_0_matmul_fanout(outs, ws, strides, 2,
                                                   *mtp_norm_batch_, batch, mtp_h_stride));
    }
    require_status(backend_.silu_mul_n(*mtp_ffn_mid_batch_,
                                       *mtp_ffn_gate_batch_,
                                       *mtp_ffn_up_batch_,
                                       static_cast<uint64_t>(batch) * ffn_stride));
    require_status(backend_.q8_0_matmul_add(*mtp_h_batch_,
                                            *mtp_h_batch_,
                                            *mtp_ffn_out_batch_,
                                            *layer.ffn_down,
                                            *mtp_ffn_mid_batch_,
                                            batch, ffn_stride, mtp_h_stride));
    record(report, "mtp.ffn_batch");

    require_status(backend_.end());
    require_status(backend_.copy_d2d(*mtp_prefix_h_, *h_batch_,
                                     static_cast<uint64_t>(batch - 1) * h_stride,
                                     h_stride));
    mtp_prefix_len_ = std::max<uint32_t>(mtp_prefix_len_, base_position + batch);
    report.ok = true;
    return report;
}

} // namespace qw3
