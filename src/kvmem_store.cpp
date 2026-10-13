#include "qw3/kvmem_store.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace qw3 {

namespace {

uint32_t ceil_div_u32(uint32_t value, uint32_t divisor) {
    if (value == 0) return 0;
    return static_cast<uint32_t>(
        (static_cast<uint64_t>(value) + divisor - 1) / divisor);
}

uint32_t clamp_u32(uint32_t value, uint32_t lo, uint32_t hi) {
    return std::min(std::max(value, lo), hi);
}

uint32_t ceil_percent(uint32_t value, uint32_t percent) {
    return static_cast<uint32_t>(
        (static_cast<uint64_t>(value) * percent + 99u) / 100u);
}

}  // namespace

std::vector<KvMemStoragePageSpan> kvmem_storage_page_spans(
        const std::vector<KvMemStorageSpan> &spans, uint32_t page_size,
        uint32_t window_begin) {
    if (!page_size) throw std::invalid_argument("KVMem page size is zero");
    std::vector<KvMemStoragePageSpan> out;
    out.reserve(spans.size());
    uint64_t window = window_begin;
    uint64_t history = spans.empty() ? 0 : spans.front().history_begin;
    for (const auto &span : spans) {
        if (!span.token_count || span.history_begin != history ||
            history > UINT64_MAX - span.token_count ||
            uint64_t(span.storage_begin) + span.token_count > INT32_MAX ||
            window + span.token_count > INT32_MAX ||
            span.storage_begin % page_size != window % page_size ||
            (!out.empty() && window % page_size != 0))
            throw std::invalid_argument("KVMem storage/window page alignment or range mismatch");
        const uint32_t first = span.storage_begin / page_size;
        const uint32_t count = static_cast<uint32_t>(
            (uint64_t(span.storage_begin) + span.token_count + page_size - 1) / page_size) - first;
        const uint32_t win_page = static_cast<uint32_t>(window / page_size);
        if (!out.empty() && out.back().storage_page + out.back().page_count == first &&
            out.back().window_page + out.back().page_count == win_page)
            out.back().page_count += count;
        else
            out.push_back({first, win_page, count});
        history += span.token_count;
        window += span.token_count;
    }
    return out;
}

KvMemStore::KvMemStore(KvMemStoreConfig cfg, std::shared_ptr<KvMemRepository> repository)
    : repository_(std::move(repository)), cfg_(cfg) {
    if (!repository_ || !cfg.block_tokens || repository_->block_tokens() != cfg.block_tokens ||
        (cfg.repository_blocks && cfg.repository_blocks != repository_->capacity()) ||
        uint64_t(repository_->capacity()) * cfg.block_tokens > INT32_MAX)
        throw std::invalid_argument("invalid bounded KVMem physical address capacity");
    if (cfg.gc.max_live_blocks && !cfg.repository_blocks)
        throw std::invalid_argument("legacy GC cannot own a reusable KVMem slot directory");
    blocks_.resize(repository_->capacity());
    repository_resident_leases_.resize(repository_->capacity());
    repository_gc_flags_.resize(repository_->capacity());
    repository_history_nodes_.reserve(repository_->capacity());
    for (uint32_t i = 0; i < block_count(); ++i) blocks_[i].block_id = i;
    // Each physical slot owns one map node, whether installed or free. Later
    // recycling extracts/reinserts that node without allocating a history log.
    for (uint32_t i = 0; i < block_count(); ++i) {
        auto it = repository_history_.emplace(i, KvMemBlockHandle{}).first;
        repository_history_nodes_.push_back(repository_history_.extract(it));
    }
}

void KvMemStore::attach_gc(std::shared_ptr<KvMemGc> gc) {
    if (repository_ && gc)
        throw std::invalid_argument("legacy GC cannot own a reusable KVMem slot directory");
    gc_ = std::move(gc);
    if (gc_) gc_->set_recent_blocks(cfg_.recent_blocks);
    gc_view_ = gc_ ? gc_->read_view() : nullptr;
}

void KvMemStore::attach_repository_gc(std::shared_ptr<KvMemRepositoryGc> gc) {
    require_no_repository_append();
    if (gc && (!repository_ || gc->repository() != repository_))
        throw std::invalid_argument("KVMem GC belongs to a different repository");
    if (repository_gc_ && repository_gc_ != gc) repository_gc_->stop();
    repository_gc_ = std::move(gc);
    if (!repository_gc_) return;
    repository_gc_->set_history_end(total_tokens_);
    for (uint32_t i = 0; i < block_count(); ++i) repository_sync_gc(i);
}

void KvMemStore::set_repository_gc_flags(KvMemBlockHandle h, uint32_t mask, uint32_t values) {
    auto lease = validate_repository_mutation(h.slot, h.generation);
    if (!lease) throw std::logic_error("KVMem repository GC flags require bounded mode");
    constexpr uint32_t allowed = KvMemGc::RawBacked | KvMemGc::ValueBacked | KvMemRepositoryGc::Indexed |
        KvMemGc::Anchor | KvMemGc::Mandatory | KvMemGc::Duplicate | KvMemGc::Superseded;
    if (mask & ~allowed) throw std::invalid_argument("KVMem automatic GC protection flags cannot be overridden");
    auto &flags = repository_gc_flags_[h.slot];
    flags = (flags & ~mask) | (values & mask);
    repository_sync_gc(h.slot);
}

void KvMemStore::repository_sync_gc(uint32_t slot, bool used) {
    if (!repository_gc_) return;
    auto lease = repository_->try_pin(repository_handle(slot));
    if (!lease) return;
    const auto &b = blocks_[slot];
    const uint32_t flags = repository_gc_flags_[slot] | KvMemGc::Registered |
        (b.n_tokens == cfg_.block_tokens ? KvMemGc::Sealed : 0u) |
        (b.tier == KvTier::GPU ? KvMemGc::Gpu : 0u) | (b.in_flight ? KvMemGc::Io : 0u) |
        (b.in_working_set || sink_block(slot) ? KvMemGc::Mandatory : 0u);
    repository_gc_->publish(lease, flags, used);
}

KvMemBlockHandle KvMemStore::repository_handle(uint32_t slot) const {
    if (!repository_ || slot >= block_count()) return {};
    return {slot, blocks_[slot].repository_generation};
}

bool KvMemStore::is_live(uint32_t id) const {
    if (id >= block_count()) return false;
    if (repository_) return bool(repository_->try_pin(repository_handle(id)));
    return !gc_ || (gc_view_ && gc_->is_live(id, gc_view_));
}

bool KvMemStore::earlier(uint32_t a, uint32_t b) const {
    if (!repository_) return a < b;
    const auto &x = blocks_[a], &y = blocks_[b];
    if (x.history_pos_start != y.history_pos_start) return x.history_pos_start < y.history_pos_start;
    return x.logical_id < y.logical_id;
}

bool KvMemStore::sink_block(uint32_t id) const {
    return repository_ ? blocks_[id].history_pos_start < uint64_t(cfg_.sink_blocks) * cfg_.block_tokens
                       : id < cfg_.sink_blocks;
}

std::vector<uint32_t> KvMemStore::chronological_ids() const {
    std::vector<uint32_t> ids;
    ids.reserve(block_count());
    if (repository_) {
        for (const auto &[begin, h] : repository_history_) {
            (void)begin;
            if (repository_->try_pin(h)) ids.push_back(h.slot);
        }
    } else {
        for (uint32_t id = 0; id < block_count(); ++id) ids.push_back(id);
    }
    return ids;
}

void KvMemStore::register_repository_block(KvMemRepository::Lease lease, uint32_t baked_pos) {
    require_no_repository_append();
    const uint32_t valid = lease ? lease.record().token_count : 0;
    install_repository_block(std::move(lease), baked_pos, valid);
}

void KvMemStore::require_no_repository_append() const {
    if (repository_append_pending_)
        throw std::logic_error("KVMem store mutation during an unfinished append");
}

void KvMemStore::install_repository_block(KvMemRepository::Lease lease, uint32_t baked_pos,
                                         uint32_t valid_tokens) {
    if (!repository_ || !lease)
        throw std::invalid_argument("bounded KVMem append requires a repository lease");
    auto verified = repository_->try_pin(lease.handle());
    if (!verified || &verified.record() != &lease.record())
        throw std::invalid_argument("KVMem block lease is stale or belongs to another repository");
    const auto &r = lease.record();
    if (!r.token_count || r.token_count > cfg_.block_tokens || r.token_begin % cfg_.block_tokens ||
        !valid_tokens || valid_tokens > r.token_count || uint64_t(baked_pos) + valid_tokens > INT32_MAX)
        throw std::invalid_argument("invalid bounded KVMem token range or local RoPE position");
    const uint32_t slot = r.handle.slot;
    if (repository_resident_leases_[slot] ||
        blocks_[slot].repository_generation == r.handle.generation)
        throw std::logic_error("KVMem repository slot already installed");
    if (repository_history_.count(r.token_begin))
        throw std::logic_error("KVMem history position already installed");
    auto &node = repository_history_nodes_[slot];
    if (node.empty()) node = repository_history_.extract(blocks_[slot].history_pos_start);
    if (node.empty()) throw std::logic_error("KVMem repository directory node missing");
    node.key() = r.token_begin;
    node.mapped() = r.handle;
    repository_history_.insert(std::move(node));
    KvMemBlock b;
    b.block_id = slot;
    b.orig_pos_start = slot * cfg_.block_tokens;
    b.n_tokens = valid_tokens;
    b.dirty_gpu = true;
    b.baked_pos = baked_pos;
    b.repository_generation = r.handle.generation;
    b.logical_id = r.logical_id;
    b.history_pos_start = r.token_begin;
    blocks_[slot] = b;
    repository_gc_flags_[slot] = 0; // never inherit backing/index validity from the old generation
    total_tokens_ = std::max(total_tokens_, r.token_begin + valid_tokens);
    repository_resident_leases_[slot] = std::move(lease);
    if (repository_gc_) repository_gc_->set_history_end(total_tokens_);
    repository_sync_gc(slot);
}

std::vector<KvMemRepository::Lease> KvMemStore::repository_read_leases() const {
    std::vector<KvMemRepository::Lease> leases;
    if (!repository_) return leases;
    leases.reserve(block_count());
    for (const auto &[begin, h] : repository_history_) {
        (void)begin;
        auto lease = repository_->try_pin(h);
        if (lease) leases.push_back(std::move(lease));
    }
    return leases;
}

KvMemStorageRead KvMemStore::repository_storage_read(uint64_t begin, uint32_t count) const {
    if (!repository_) throw std::logic_error("KVMem storage mapping requires bounded repository");
    if (begin > UINT64_MAX - count) throw std::overflow_error("KVMem history interval overflow");
    KvMemStorageRead out;
    const uint64_t end = begin + count;
    if (count == 0) return out;
    auto it = repository_history_.upper_bound(begin);
    if (it != repository_history_.begin()) --it;
    uint64_t covered = begin;
    while (covered < end && it != repository_history_.end()) {
        const auto [first, h] = *it++;
        auto lease = repository_->try_pin(h);
        if (!lease) break;
        const auto &b = blocks_[h.slot];
        const uint64_t last = first + b.n_tokens;
        if (first > covered || last <= covered) break;
        const uint32_t take = static_cast<uint32_t>(std::min(last, end) - covered);
        out.spans.push_back({covered, b.orig_pos_start + static_cast<uint32_t>(covered - first), take, h});
        out.leases.push_back(std::move(lease));
        covered += take;
    }
    if (covered != end)
        throw std::runtime_error("KVMem required-history-deleted: missing or uncommitted storage interval");
    return out;
}

KvMemStore::RepositoryAppend::~RepositoryAppend() {
    if (owner_) owner_->repository_append_pending_ = false;
    // Reservation destructors retire unpublished bundles; outstanding IO
    // clones keep them alive until actual completion before CPU reclaim.
}

std::vector<KvMemRepository::Lease> KvMemStore::RepositoryAppend::io_leases() const {
    if (!owner_) throw std::logic_error("KVMem append already finished");
    std::vector<KvMemRepository::Lease> leases;
    leases.reserve(reservations_.size() + (tail_ ? 1 : 0));
    if (tail_) leases.push_back(tail_.clone());
    for (const auto &r : reservations_) leases.push_back(r.lease());
    return leases;
}

bool KvMemStore::RepositoryAppend::extend(uint32_t count) {
    if (!owner_) throw std::logic_error("KVMem append already finished");
    if (count <= count_) return true;
    auto &s = *owner_;
    const uint32_t B = s.cfg_.block_tokens;
    if (begin_ > UINT64_MAX - count || uint64_t(baked_begin_) + count > INT32_MAX)
        throw std::invalid_argument("KVMem append extension interval overflow");
    const uint64_t new_end = begin_ + count;
    if ((new_end - 1) / B > (UINT64_MAX - B) / B)
        throw std::overflow_error("KVMem append extension block overflow");
    const uint64_t old_reserved_end = ((end() - 1) / B + 1) * B;
    const uint32_t needed = new_end <= old_reserved_end ? 0 :
        ceil_div_u32(static_cast<uint32_t>(new_end - old_reserved_end), B);
    if (s.repository_->ready_slots() < needed) return false;
    spans_.reserve(spans_.size() + needed);
    reservations_.reserve(reservations_.size() + needed);
    const size_t previous_spans = spans_.size();
    for (uint64_t pos = old_reserved_end; pos < new_end; pos += B) {
        auto r = s.repository_->try_reserve(pos, B);
        if (!r) throw std::logic_error("KVMem append producer lost ready capacity");
        const auto h = r.record().handle;
        spans_.push_back({pos, h.slot * B, static_cast<uint32_t>(std::min<uint64_t>(B, new_end - pos)), h});
        reservations_.push_back(std::move(r));
    }
    auto &last = spans_[previous_spans - 1];
    last.token_count = static_cast<uint32_t>(std::min(new_end, old_reserved_end) - last.history_begin);
    count_ = count;
    return true;
}

KvMemStorageRead KvMemStore::RepositoryAppend::storage_write(uint64_t begin, uint32_t count) const {
    if (!owner_ || begin < begin_ || begin > end() || count > end() - begin)
        throw std::invalid_argument("KVMem write view outside prepared append");
    KvMemStorageRead out;
    if (!count) return out;
    const uint64_t stop = begin + count;
    for (size_t i = 0; i < spans_.size(); ++i) {
        const auto &span = spans_[i];
        const uint64_t first = std::max(begin, span.history_begin);
        const uint64_t last = std::min(stop, span.history_begin + span.token_count);
        if (first >= last) continue;
        out.spans.push_back({first, span.storage_begin + static_cast<uint32_t>(first - span.history_begin),
                             static_cast<uint32_t>(last - first), span.handle});
        out.leases.push_back(tail_ && i == 0 ? tail_.clone() : reservations_[i - (tail_ ? 1 : 0)].lease());
    }
    return out;
}

std::vector<std::pair<uint32_t, uint32_t>> KvMemStore::RepositoryAppend::discarded_page_ranges(
        uint32_t accepted, uint32_t page_size) const {
    if (!owner_ || accepted > count_ || !page_size || owner_->cfg_.block_tokens % page_size)
        throw std::invalid_argument("invalid KVMem speculative page cleanup");
    const uint64_t kept_end = begin_ + accepted;
    std::vector<std::pair<uint32_t, uint32_t>> out;
    for (const auto &span : spans_) {
        const uint64_t end = span.history_begin + span.token_count;
        if (end <= kept_end) continue;
        const uint64_t storage_end = uint64_t(span.storage_begin) + span.token_count;
        const uint64_t storage_begin = uint64_t(span.storage_begin) +
            (kept_end > span.history_begin ? kept_end - span.history_begin : 0);
        const uint32_t first = static_cast<uint32_t>((storage_begin + page_size - 1) / page_size);
        const uint32_t last = static_cast<uint32_t>((storage_end + page_size - 1) / page_size);
        if (last <= first) continue;
        if (!out.empty() && uint64_t(out.back().first) + out.back().second == first)
            out.back().second += last - first;
        else out.emplace_back(first, last - first);
    }
    return out;
}

std::unique_ptr<KvMemStore::RepositoryAppend> KvMemStore::prepare_repository_append(
        uint32_t count, uint32_t baked_begin) {
    require_no_repository_append();
    if (!repository_ || !count || total_tokens_ > UINT64_MAX - count ||
        uint64_t(baked_begin) + count > INT32_MAX)
        throw std::invalid_argument("invalid KVMem append history/local interval");
    const uint32_t B = cfg_.block_tokens;
    const uint64_t begin = total_tokens_, end = begin + count;
    const uint32_t tail_offset = static_cast<uint32_t>(begin % B);
    const uint32_t tail_take = tail_offset ? std::min(count, B - tail_offset) : 0;
    const uint32_t needed = ceil_div_u32(count - tail_take, B);
    // Leave room for the immutable full-block extent of the final reservation.
    if (needed && (end - 1) / B > (UINT64_MAX - B) / B)
        throw std::overflow_error("KVMem reserved block interval overflow");
    if (repository_->ready_slots() < needed) return nullptr;
    auto append = std::unique_ptr<RepositoryAppend>(new RepositoryAppend(*this));
    append->begin_ = begin;
    append->count_ = count;
    append->baked_begin_ = baked_begin;
    append->spans_.reserve(needed + (tail_take ? 1 : 0));
    append->reservations_.reserve(needed);
    if (tail_take) {
        auto it = repository_history_.find(begin - tail_offset);
        if (it == repository_history_.end() || !(append->tail_ = repository_->try_pin(it->second)))
            throw std::runtime_error("KVMem append tail unavailable");
        const auto &r = append->tail_.record();
        const auto &b = blocks_[r.handle.slot];
        if (b.n_tokens != tail_offset || b.tier != KvTier::GPU ||
            b.baked_pos + tail_offset != baked_begin || r.token_count < tail_offset + tail_take)
            throw std::logic_error("KVMem append tail extent/residency/RoPE mismatch");
        append->spans_.push_back({begin, b.orig_pos_start + tail_offset, tail_take, r.handle});
    }
    for (uint64_t pos = begin + tail_take; pos < end; pos += B) {
        auto reservation = repository_->try_reserve(pos, B);
        if (!reservation) throw std::logic_error("KVMem ready capacity changed on single append producer");
        const auto h = reservation.record().handle;
        append->spans_.push_back({pos, h.slot * B, static_cast<uint32_t>(std::min<uint64_t>(B, end - pos)), h});
        append->reservations_.push_back(std::move(reservation));
    }
    repository_append_pending_ = true;
    return append;
}

void KvMemStore::RepositoryAppend::commit(uint32_t accepted) {
    if (!owner_ || accepted > count_) throw std::invalid_argument("invalid KVMem append commit");
    auto &s = *owner_;
    if (s.total_tokens_ != begin_) throw std::logic_error("KVMem history changed during append");
    const uint64_t end = begin_ + accepted;
    size_t index = 0;
    if (tail_) {
        const auto &span = spans_[index++];
        auto &b = s.blocks_[span.handle.slot];
        b.n_tokens += std::min(span.token_count, accepted);
        if (accepted) { b.dirty_gpu = true; b.ssd_clean = false; }
        if (accepted) {
            s.repository_gc_flags_[span.handle.slot] &=
                ~(KvMemGc::RawBacked | KvMemGc::ValueBacked | KvMemRepositoryGc::Indexed);
            s.repository_sync_gc(span.handle.slot);
        }
    }
    for (auto &r : reservations_) {
        const auto &span = spans_[index++];
        if (span.history_begin >= end) continue;
        const uint32_t valid = static_cast<uint32_t>(std::min<uint64_t>(span.token_count, end - span.history_begin));
        s.install_repository_block(r.commit(),
            baked_begin_ + static_cast<uint32_t>(span.history_begin - begin_), valid);
    }
    s.total_tokens_ = end;
    if (s.repository_gc_) s.repository_gc_->set_history_end(end);
    s.repository_append_pending_ = false;
    owner_ = nullptr;
    reservations_.clear(); // cancelled MTP suffix returns via the CPU reclaimer
    tail_.reset();
}

KvMemStorageRead KvMemStore::repository_slot_read(uint32_t first, uint32_t count,
                                                 const RepositoryAppend *prepared) const {
    if (!repository_ || first > block_count() || count > block_count() - first)
        throw std::invalid_argument("KVMem physical read outside bounded directory");
    if (prepared && prepared->owner_ != this)
        throw std::invalid_argument("KVMem physical read has a foreign/completed reservation");
    auto writes = prepared ? prepared->io_leases() : std::vector<KvMemRepository::Lease>{};
    std::sort(writes.begin(), writes.end(), [](const auto &a, const auto &b) {
        return a.handle().slot < b.handle().slot;
    });
    std::vector<KvMemStorageSpan> spans = prepared ? prepared->spans() : std::vector<KvMemStorageSpan>{};
    std::sort(spans.begin(), spans.end(), [](const auto &a, const auto &b) {
        return a.handle.slot < b.handle.slot;
    });
    KvMemStorageRead out;
    out.spans.reserve(count); out.leases.reserve(count);
    auto write = std::lower_bound(writes.begin(), writes.end(), first,
        [](const auto &a, uint32_t b) { return a.handle().slot < b; });
    auto span = std::lower_bound(spans.begin(), spans.end(), first,
        [](const auto &a, uint32_t b) { return a.handle.slot < b; });
    for (uint32_t id = first; id < first + count; ++id) {
        KvMemRepository::Lease lease;
        uint32_t valid = 0;
        if (write != writes.end() && write->handle().slot == id) {
            if (span == spans.end() || span->handle != write->handle())
                throw std::logic_error("KVMem prepared physical read lost its extent");
            valid = span->storage_begin % cfg_.block_tokens + span->token_count;
            lease = std::move(*write++); ++span;
        } else {
            lease = repository_->try_pin(repository_handle(id));
            if (!lease) continue;
            valid = blocks_[id].n_tokens;
        }
        if (!valid || valid > cfg_.block_tokens)
            throw std::logic_error("KVMem physical read has invalid token coverage");
        out.spans.push_back({lease.record().token_begin, id * cfg_.block_tokens, valid, lease.handle()});
        out.leases.push_back(std::move(lease));
    }
    return out;
}

std::vector<uint32_t> KvMemStore::history_span_blocks(uint64_t begin, uint64_t end,
                                                     bool require_complete) const {
    std::vector<uint32_t> ids;
    if (end <= begin) return ids;
    uint64_t covered = begin;
    auto visit = [&](uint32_t id) {
        if (!is_live(id)) return;
        const auto &b = blocks_[id];
        const uint64_t first = repository_ ? b.history_pos_start : b.orig_pos_start;
        const uint64_t last = first + b.n_tokens;
        if (last <= begin || first >= end) return;
        if (require_complete && first > covered)
            throw std::runtime_error("KVMem required-history-deleted: missing token interval");
        ids.push_back(id);
        covered = std::max(covered, std::min(end, last));
    };
    if (repository_) {
        auto it = repository_history_.upper_bound(begin);
        if (it != repository_history_.begin()) --it;
        for (; it != repository_history_.end() && it->first < end; ++it) visit(it->second.slot);
    } else {
        for (const auto id : chronological_ids()) visit(id);
    }
    if (require_complete && covered < end)
        throw std::runtime_error("KVMem required-history-deleted: missing token suffix");
    return ids;
}

std::vector<KvMemRepository::Lease> KvMemStore::repository_checkpoint_leases() const {
    std::vector<KvMemRepository::Lease> leases;
    if (!repository_) return leases;
    for (const auto &b : blocks_) {
        if (!b.repository_generation || !b.n_tokens || (!b.in_working_set && b.tier != KvTier::GPU)) continue;
        auto lease = repository_->try_pin(repository_handle(b.block_id));
        if (!lease) throw std::runtime_error("KVMem checkpoint requires a missing repository block");
        leases.push_back(std::move(lease));
    }
    return leases;
}

std::shared_ptr<KvMemStore::RepositoryCheckpoint> KvMemStore::repository_checkpoint(uint64_t position,
                                                                                bool pin_working_set) {
    if (!repository_ || position > total_tokens_)
        throw std::invalid_argument("KVMem checkpoint exceeds committed repository history");
    auto free = std::find_if(repository_checkpoints_.begin(), repository_checkpoints_.end(),
        [](const auto &p) { return p.expired(); });
    if (free == repository_checkpoints_.end())
        throw std::runtime_error("KVMem bounded checkpoint capacity exhausted");
    auto snapshot = std::make_shared<RepositoryCheckpoint>();
    snapshot->repository_ = repository_;
    snapshot->position_ = position;
    if (pin_working_set) protect_repository_checkpoint(snapshot);
    *free = snapshot;
    return snapshot;
}

void KvMemStore::protect_repository_checkpoint(const std::shared_ptr<RepositoryCheckpoint> &snapshot) {
    if (!repository_checkpoint_valid(snapshot))
        throw std::runtime_error("KVMem cannot pin an invalid checkpoint");
    auto leases = repository_checkpoint_leases();
    leases.erase(std::remove_if(leases.begin(), leases.end(), [&](const auto &lease) {
        return lease.record().token_begin >= snapshot->position_;
    }), leases.end());
    snapshot->leases_ = std::move(leases);
}

bool KvMemStore::repository_checkpoint_valid(const std::shared_ptr<RepositoryCheckpoint> &s) const {
    if (!repository_ || !s || !s->valid_ || s->repository_ != repository_ ||
        s->position_ > total_tokens_) return false;
    for (const auto &lease : s->leases_) {
        const auto h = lease.handle();
        if (repository_handle(h.slot) != h || !repository_->try_pin(h)) return false;
    }
    return true;
}

KvMemStore::RepositoryRewind::~RepositoryRewind() {
    if (owner_) owner_->repository_append_pending_ = false;
}

std::unique_ptr<KvMemStore::RepositoryRewind> KvMemStore::prepare_repository_rewind(uint64_t position) {
    require_no_repository_append();
    if (!repository_ || position > total_tokens_)
        throw std::invalid_argument("KVMem rewind exceeds committed history");
    auto rewind = std::unique_ptr<RepositoryRewind>(new RepositoryRewind(*this));
    rewind->position_ = position;
    if (position != total_tokens_) {
        const auto B = cfg_.block_tokens;
        const uint64_t boundary = position - position % B;
        auto it = repository_history_.lower_bound(boundary);
        if (position % B && (it == repository_history_.end() || it->first != boundary ||
                !repository_->try_pin(it->second)))
            throw std::runtime_error("KVMem rewind partial tail has been deleted");
        for (; it != repository_history_.end(); ++it) {
            const auto h = it->second;
            auto lease = repository_->try_pin(h);
            if (!lease) continue; // historical GC holes remain holes
            const auto &b = blocks_[h.slot];
            if (b.in_flight)
                throw std::logic_error("KVMem rewind requires completed KV transfers");
            const uint32_t keep = b.history_pos_start < position
                ? static_cast<uint32_t>(position - b.history_pos_start) : 0;
            if (keep > b.n_tokens)
                throw std::logic_error("KVMem rewind partial tail extent mismatch");
            if (keep == b.n_tokens) continue;
            rewind->captured_.spans.push_back({b.history_pos_start, b.orig_pos_start, b.n_tokens, h});
            rewind->captured_.leases.push_back(std::move(lease));
            rewind->discarded_.push_back({b.history_pos_start + keep, b.orig_pos_start + keep,
                                         b.n_tokens - keep, h});
        }
    }
    repository_append_pending_ = true;
    return rewind;
}

void KvMemStore::RepositoryRewind::commit() {
    if (!owner_) throw std::logic_error("KVMem rewind already finished");
    auto &s = *owner_;
    // Do this before exposing the shorter tail, and never revive these objects
    // when another branch grows back through the discarded checkpoint position.
    for (auto &entry : s.repository_checkpoints_) {
        if (auto checkpoint = entry.lock(); checkpoint && checkpoint->position_ > position_) {
            checkpoint->valid_ = false;
            checkpoint->leases_.clear();
        }
    }
    for (const auto &span : captured_.spans) {
        auto &b = s.blocks_[span.handle.slot];
        if (b.history_pos_start < position_) {
            b.n_tokens = static_cast<uint32_t>(position_ - b.history_pos_start);
            b.dirty_gpu = true;
            b.ssd_clean = false;
            s.repository_gc_flags_[b.block_id] &=
                ~(KvMemGc::RawBacked | KvMemGc::ValueBacked | KvMemRepositoryGc::Indexed);
            s.repository_sync_gc(b.block_id);
        } else {
            // captured_ still pins the payload through foreground page cleanup.
            s.repository_->retire(span.handle);
            s.repository_resident_leases_[b.block_id].reset();
            b.n_tokens = 0;
            b.in_working_set = false;
            s.repository_gc_flags_[b.block_id] = 0;
        }
    }
    // Also remove stale directory nodes whose payload GC already reclaimed.
    // Their storage generations are not touched; the physical slot may belong
    // to a later lease until that generation's ordinary reclaim completes.
    auto it = s.repository_history_.lower_bound(position_);
    while (it != s.repository_history_.end()) {
        const auto current = it++;
        const auto slot = current->second.slot;
        s.repository_history_nodes_[slot] = s.repository_history_.extract(current);
    }
    s.total_tokens_ = position_;
    s.repository_scores_pending_ = false;
    s.repository_score_handles_.clear();
    if (s.repository_gc_) s.repository_gc_->set_history_end(position_);
    s.repository_append_pending_ = false;
    owner_ = nullptr;
}

KvMemRepositoryScoreFrame KvMemStore::repository_score_frame(
        const std::vector<KvMemIndexStamp> &index, uint64_t boundary, bool mask_kept) const {
    if (!repository_ || index.size() != block_count())
        throw std::invalid_argument("KVMem score index has no fixed repository directory");
    KvMemRepositoryScoreFrame frame;
    frame.leases = repository_read_leases();
    frame.enabled.assign(block_count(), 0);
    std::vector<uint32_t> chronological;
    chronological.reserve(frame.leases.size());
    for (const auto &lease : frame.leases) {
        const auto h = lease.handle();
        const auto &b = blocks_[h.slot];
        if (b.history_pos_start >= boundary) continue;
        const uint32_t expected = static_cast<uint32_t>(
            std::min<uint64_t>(b.n_tokens, boundary - b.history_pos_start));
        if (!expected || index[h.slot].handle != h || index[h.slot].tokens != expected)
            throw std::runtime_error("KVMem index generation or token coverage mismatch");
        frame.enabled[h.slot] = 1;
        chronological.push_back(h.slot);
    }
    // Slot reuse destroys the numerical ordering of sink/recent bands. Mask
    // their historical order, never [0,sink) / [capacity-recent,capacity).
    const auto n = chronological.size();
    const auto sink = std::min<size_t>(cfg_.sink_blocks, n);
    const auto recent = std::min<size_t>(cfg_.recent_blocks, n);
    if (mask_kept && n > budget_blocks() && sink + recent < n) {
        for (size_t i = 0; i < sink; ++i) frame.enabled[chronological[i]] = 0;
        for (size_t i = n - recent; i < n; ++i) frame.enabled[chronological[i]] = 0;
    }
    return frame;
}

void KvMemStore::set_repository_retrieval_scores(
        const std::vector<KvMemBlockHandle> &handles, const std::vector<double> &scores) {
    require_no_repository_append();
    if (!repository_ || handles.size() != scores.size())
        throw std::invalid_argument("invalid bounded KVMem score frame");
    std::vector<KvMemRepository::Lease> leases;
    leases.reserve(handles.size());
    std::vector<uint8_t> seen(block_count(), 0);
    for (const auto h : handles) {
        if (h != repository_handle(h.slot) || !h.generation)
            throw std::runtime_error("KVMem score frame contains a stale block generation");
        auto l = repository_->try_pin(h);
        if (!l || seen[h.slot]) throw std::runtime_error("KVMem score frame has missing or duplicate blocks");
        seen[h.slot] = 1; leases.push_back(std::move(l));
    }
    for (auto &b : blocks_) b.attn_score = b.retrieval_score = 0;
    for (size_t i = 0; i < handles.size(); ++i) {
        auto &b = blocks_[handles[i].slot];
        b.attn_score = b.retrieval_score = scores[i];
    }
    repository_score_handles_ = handles;
    repository_scores_pending_ = true;
}

void KvMemStore::accumulate_repository_attn(
        const std::vector<KvMemBlockHandle> &handles, const std::vector<double> &scores) {
    require_no_repository_append();
    if (!repository_ || handles.size() != scores.size())
        throw std::invalid_argument("invalid bounded KVMem attention frame");
    std::vector<KvMemRepository::Lease> leases;
    leases.reserve(handles.size());
    std::vector<uint8_t> seen(block_count(), 0);
    for (auto h : handles) {
        auto lease = validate_repository_mutation(h.slot, h.generation);
        if (seen[h.slot]) throw std::invalid_argument("duplicate KVMem attention block");
        seen[h.slot] = 1; leases.push_back(std::move(lease));
    }
    // Validate the entire original window before modifying any heat. The
    // accumulator belongs to these generations, never their replacement slots.
    for (size_t i = 0; i < handles.size(); ++i) {
        auto &b = blocks_[handles[i].slot];
        b.profile_score += scores[i];
        b.attn_score = b.profile_score;
    }
}

void KvMemStore::set_runtime_select_budget(uint32_t tokens) {
    if (tokens == 0) {
        runtime_select_budget_ = 0;
        return;
    }
    if (tokens > cfg_.select_budget) {
        throw std::invalid_argument(
            "KVMem request semantic budget exceeds configured maximum");
    }
    if (tokens < cfg_.block_tokens || tokens % cfg_.block_tokens != 0) {
        throw std::invalid_argument(
            "KVMem request semantic budget must be a positive multiple of "
            "block_tokens");
    }
    const uint32_t blocks = tokens / cfg_.block_tokens;
    if (static_cast<uint64_t>(cfg_.sink_blocks) + cfg_.recent_blocks >
        blocks) {
        throw std::invalid_argument(
            "KVMem request semantic budget is smaller than the configured "
            "sink + recent allocation");
    }
    runtime_select_budget_ = tokens;
}

KvMemKeepAllocation resolve_kvmem_keep_allocation(
        uint32_t block_tokens,
        uint32_t select_budget,
        int64_t sink_blocks,
        int64_t recent_blocks,
        int64_t sink_tokens,
        int64_t recent_tokens) {
    if (block_tokens == 0) {
        throw std::runtime_error(
            "KVMem keep allocation requires block_tokens > 0");
    }
    if (select_budget < block_tokens) {
        throw std::runtime_error(
            "KVMem selection budget must contain at least one block");
    }
    if (sink_blocks >= 0 && sink_tokens >= 0) {
        throw std::runtime_error(
            "KVMem sink allocation cannot specify both blocks and tokens");
    }
    if (recent_blocks >= 0 && recent_tokens >= 0) {
        throw std::runtime_error(
            "KVMem recent allocation cannot specify both blocks and tokens");
    }

    KvMemKeepAllocation out;
    const uint32_t budget_blocks = select_budget / block_tokens;

    auto resolve_band = [&](int64_t explicit_blocks,
                            int64_t explicit_tokens,
                            uint32_t auto_tokens,
                            const char *name,
                            uint32_t &target_tokens,
                            uint32_t &blocks,
                            uint32_t &effective_tokens,
                            KvMemKeepSource &source) {
        if (explicit_blocks >= 0) {
            if (static_cast<uint64_t>(explicit_blocks) >
                std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error(
                    std::string("KVMem ") + name +
                    " block allocation is too large");
            }
            blocks = static_cast<uint32_t>(explicit_blocks);
            const uint64_t tokens =
                static_cast<uint64_t>(blocks) * block_tokens;
            if (tokens > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error(
                    std::string("KVMem ") + name +
                    " block allocation overflows token accounting");
            }
            target_tokens = static_cast<uint32_t>(tokens);
            effective_tokens = target_tokens;
            source = KvMemKeepSource::Blocks;
            return;
        }

        const uint64_t requested = explicit_tokens >= 0
            ? static_cast<uint64_t>(explicit_tokens)
            : static_cast<uint64_t>(auto_tokens);
        if (requested > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error(
                std::string("KVMem ") + name +
                " token allocation is too large");
        }
        target_tokens = static_cast<uint32_t>(requested);
        blocks = ceil_div_u32(target_tokens, block_tokens);
        const uint64_t rounded_tokens =
            static_cast<uint64_t>(blocks) * block_tokens;
        if (rounded_tokens > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error(
                std::string("KVMem ") + name +
                " rounded token allocation overflows accounting");
        }
        effective_tokens = static_cast<uint32_t>(rounded_tokens);
        source = explicit_tokens >= 0
            ? KvMemKeepSource::Tokens
            : KvMemKeepSource::Auto;
    };

    const uint32_t auto_sink_tokens =
        clamp_u32(ceil_percent(select_budget, 1), 1024, 2048);
    const uint32_t auto_recent_tokens =
        clamp_u32(ceil_percent(select_budget, 8), 4096, 16384);
    resolve_band(
        sink_blocks, sink_tokens, auto_sink_tokens, "sink",
        out.sink_target_tokens, out.sink_blocks,
        out.sink_effective_tokens, out.sink_source);
    resolve_band(
        recent_blocks, recent_tokens, auto_recent_tokens, "recent",
        out.recent_target_tokens, out.recent_blocks,
        out.recent_effective_tokens, out.recent_source);

    if (out.sink_blocks > budget_blocks ||
        out.recent_blocks > budget_blocks ||
        static_cast<uint64_t>(out.sink_blocks) + out.recent_blocks >
            budget_blocks) {
        throw std::runtime_error(
            "KVMem sink + recent allocation exceeds the selection budget");
    }
    return out;
}

void KvMemStore::register_append(uint32_t n_new_tokens) {
    if (repository_) throw std::logic_error("bounded KVMem append requires explicit repository allocation");
    if (total_tokens_ + n_new_tokens > UINT32_MAX)
        throw std::overflow_error("legacy KVMem history position overflow");
    if (n_new_tokens == 0) return;
    const uint32_t bt = cfg_.block_tokens;

    // Extend the trailing partial block first, then spill into new blocks.
    uint32_t remaining = n_new_tokens;
    if (!blocks_.empty()) {
        KvMemBlock &last = blocks_.back();
        if (last.n_tokens < bt) {
            const uint32_t room = bt - last.n_tokens;
            const uint32_t take = std::min(room, remaining);
            last.n_tokens += take;
            remaining -= take;
            total_tokens_ += take;
            if (gc_) gc_sync_block(last.block_id);
        }
    }
    while (remaining > 0) {
        KvMemBlock b;
        b.block_id = static_cast<uint32_t>(blocks_.size());
        b.orig_pos_start = total_tokens_;
        b.n_tokens = std::min(bt, remaining);
        b.baked_pos = static_cast<int64_t>(b.orig_pos_start);  // baked at true pos after prefill
        blocks_.push_back(b);
        if (gc_) {
            gc_->register_block(b.block_id, b.n_tokens == bt);
            if (b.block_id < cfg_.sink_blocks) gc_->update_flags(b.block_id, KvMemGc::Mandatory, KvMemGc::Mandatory);
        }
        total_tokens_ += b.n_tokens;
        remaining -= b.n_tokens;
    }
}

std::vector<KvMemDroppedBlock> KvMemStore::truncate_to(uint32_t token_pos) {
    if (repository_) throw std::logic_error("bounded KVMem rewind requires a versioned repository transaction");
    std::vector<KvMemDroppedBlock> dropped;
    if (token_pos >= total_tokens_) return dropped;
    // Pop fully-past blocks from the back, then shrink the trailing partial one.
    while (!blocks_.empty() && blocks_.back().orig_pos_start >= token_pos) {
        const KvMemBlock &b = blocks_.back();
        dropped.push_back(KvMemDroppedBlock{b.block_id, b.tier,
                                            b.gpu_slot, b.cpu_slot, b.nvme_slot});
        total_tokens_ -= b.n_tokens;
        blocks_.pop_back();
    }
    if (gc_) gc_->forget_tail(block_count());
    if (!blocks_.empty()) {
        KvMemBlock &last = blocks_.back();
        if (last.orig_pos_end() > token_pos) {
            const uint32_t keep = token_pos - last.orig_pos_start;
            total_tokens_ -= (last.n_tokens - keep);
            last.n_tokens = keep;
            if (gc_) gc_sync_block(last.block_id);
        }
    }
    return dropped;
}

void KvMemStore::accumulate_attn(const std::vector<double> &scores) {
    if (repository_) throw std::logic_error("bounded KVMem scores require block generations");
    const uint32_t n = std::min<uint32_t>(static_cast<uint32_t>(scores.size()),
                                          block_count());
    for (uint32_t i = 0; i < n; ++i) {
        blocks_[i].profile_score += scores[i];
        blocks_[i].attn_score = blocks_[i].profile_score;
    }
}

void KvMemStore::set_attn_scores(const std::vector<double> &scores) {
    set_retrieval_scores(scores);
}

void KvMemStore::set_retrieval_scores(const std::vector<double> &scores) {
    if (repository_) throw std::logic_error("bounded KVMem scores require block generations");
    const uint32_t n = block_count();
    for (uint32_t i = 0; i < n; ++i) {
        blocks_[i].retrieval_score = (i < scores.size()) ? scores[i] : 0.0;
        blocks_[i].attn_score = blocks_[i].retrieval_score;
    }
}

KvMemRepository::Lease KvMemStore::validate_repository_mutation(uint32_t id, uint64_t generation) const {
    // Completion of an older batch may overlap a prepared append. Versioned
    // tier/index flags do not change the append's historical directory or
    // token extents; structural mutations still require no pending append.
    if (!repository_) return {};
    const KvMemBlockHandle h{id, generation};
    if (!generation || h != repository_handle(id))
        throw std::runtime_error("KVMem metadata update requires the current block generation");
    auto lease = repository_->try_pin(h);
    if (!lease) throw std::runtime_error("KVMem metadata update refers to a retired block");
    return lease;
}

void KvMemStore::set_block_tier(uint32_t block_id, KvTier tier,
                                int32_t cpu_slot, int32_t nvme_slot, uint64_t generation) {
    auto mutation = validate_repository_mutation(block_id, generation);
    if (block_id >= block_count()) return;
    KvMemBlock &b = blocks_[block_id];
    if (repository_) {
        if (!is_live(block_id)) throw std::runtime_error("KVMem tier change requires a live block");
        if (tier == KvTier::GPU && !repository_resident_leases_[block_id]) {
            auto lease = repository_->try_pin(repository_handle(block_id));
            if (!lease) throw std::runtime_error("KVMem stage-in block retired");
            repository_resident_leases_[block_id] = std::move(lease);
        }
        if (tier != KvTier::GPU && !b.in_working_set)
            repository_resident_leases_[block_id].reset();
    }
    if (gc_ && b.tier == KvTier::GPU && tier != KvTier::GPU) gc_->used(block_id);
    b.tier = tier;
    if (cfg_.optimize_stage_out) {
        if (tier == KvTier::GPU) {
            // GPU becomes active, but clean CPU/SSD cache copies remain valid.
            if (cpu_slot >= 0) b.cpu_slot = cpu_slot;
            if (nvme_slot >= 0) b.nvme_slot = nvme_slot;
        } else {
            b.gpu_slot = -1;
            b.cpu_slot = cpu_slot;
            if (nvme_slot >= 0) b.nvme_slot = nvme_slot;
        }
        if (repository_ && tier == KvTier::CPU && cpu_slot >= 0 && !b.in_flight) {
            b.dirty_gpu = false; // synchronous stage-out has completed
            repository_gc_flags_[block_id] |= KvMemGc::ValueBacked;
        }
        if (gc_) gc_sync_block(block_id);
        repository_sync_gc(block_id);
        return;
    }
    if (tier == KvTier::GPU) {
        b.cpu_slot = -1;
        b.nvme_slot = -1;
        b.ssd_clean = false;
    } else {
        b.gpu_slot = -1;
        b.cpu_slot = cpu_slot;
        b.nvme_slot = nvme_slot;
        b.ssd_clean = tier == KvTier::SSD && nvme_slot >= 0;
    }
    if (gc_) gc_sync_block(block_id);
    repository_sync_gc(block_id);
}

void KvMemStore::set_block_cpu_copy(uint32_t block_id, int32_t cpu_slot, uint64_t generation) {
    auto mutation = validate_repository_mutation(block_id, generation);
    if (block_id >= block_count()) return;
    blocks_[block_id].cpu_slot = cpu_slot;
    if (repository_ && cpu_slot >= 0) blocks_[block_id].dirty_gpu = false;
    if (gc_) gc_sync_block(block_id);
    if (repository_) {
        const auto &b = blocks_[block_id];
        const bool backed = !b.dirty_gpu && !b.in_flight && (cpu_slot >= 0 || b.ssd_clean);
        auto &flags = repository_gc_flags_[block_id];
        flags = (flags & ~KvMemGc::ValueBacked) | (backed ? KvMemGc::ValueBacked : 0u);
        repository_sync_gc(block_id);
    }
}

void KvMemStore::set_block_ssd_backing(uint32_t block_id, int32_t nvme_slot,
                                       bool clean, uint64_t generation) {
    auto mutation = validate_repository_mutation(block_id, generation);
    if (block_id >= block_count()) return;
    KvMemBlock &b = blocks_[block_id];
    b.nvme_slot = nvme_slot;
    b.ssd_clean = clean && nvme_slot >= 0;
    if (repository_ && b.ssd_clean) b.dirty_gpu = false;
    if (gc_) gc_sync_block(block_id);
    if (repository_) {
        const bool backed = !b.dirty_gpu && !b.in_flight && (b.cpu_slot >= 0 || b.ssd_clean);
        auto &flags = repository_gc_flags_[block_id];
        flags = (flags & ~KvMemGc::ValueBacked) | (backed ? KvMemGc::ValueBacked : 0u);
        repository_sync_gc(block_id);
    }
}

void KvMemStore::set_block_io_in_flight(uint32_t block_id, bool in_flight, uint64_t generation) {
    auto mutation = validate_repository_mutation(block_id, generation);
    if (block_id >= block_count()) return;
    blocks_[block_id].in_flight = in_flight;
    if (gc_) gc_sync_block(block_id);
    repository_sync_gc(block_id);
}

void KvMemStore::set_block_baked_pos(uint32_t block_id, int64_t baked_pos, uint64_t generation) {
    auto mutation = validate_repository_mutation(block_id, generation);
    if (block_id >= block_count()) return;
    blocks_[block_id].baked_pos = baked_pos;
}

void KvMemStore::record_block_rerope(uint32_t block_id, int64_t baked_pos, uint64_t generation) {
    auto mutation = validate_repository_mutation(block_id, generation);
    if (block_id >= block_count()) return;
    KvMemBlock &block = blocks_[block_id];
    if (block.baked_pos != baked_pos) {
        ++block.remap_count;
        const uint64_t delta = static_cast<uint64_t>(
            block.baked_pos > baked_pos ? block.baked_pos - baked_pos
                                        : baked_pos - block.baked_pos);
        block.remap_abs_delta += delta;
    }
    block.baked_pos = baked_pos;
}

std::vector<uint32_t> KvMemStore::pick_prefill_pressure_blocks() const {
    return pick_prefill_pressure_blocks({});
}

std::vector<uint32_t> KvMemStore::pick_prefill_pressure_blocks(
        const std::vector<uint32_t> &mandatory_blocks) const {
    return pick_prefill_pressure_from(chronological_ids(), mandatory_blocks);
}

KvMemStore::RepositoryPressureSelection KvMemStore::pin_repository_pressure_selection(
        const std::vector<uint32_t> &mandatory_blocks) const {
    if (!repository_) throw std::logic_error("pinned pressure selection requires bounded repository");
    RepositoryPressureSelection result;
    result.leases = repository_read_leases();
    std::vector<uint32_t> order;
    order.reserve(result.leases.size());
    for (const auto &lease : result.leases) order.push_back(lease.handle().slot);
    result.blocks = pick_prefill_pressure_from(order, mandatory_blocks);
    std::vector<uint8_t> chosen(block_count(), 0);
    for (auto id : result.blocks) chosen[id] = 1;
    result.leases.erase(std::remove_if(result.leases.begin(), result.leases.end(),
        [&](const auto &lease) { return !chosen[lease.handle().slot]; }), result.leases.end());
    return result;
}

std::vector<uint32_t> KvMemStore::pick_prefill_pressure_from(
        const std::vector<uint32_t> &order,
        const std::vector<uint32_t> &mandatory_blocks) const {
    const uint32_t n = block_count();
    std::vector<uint32_t> selected;
    if (n == 0) return selected;
    gc_validate_mandatory(mandatory_blocks);
    // A worker can rescue a transiently retired record after the snapshot.
    // Such a newly visible slot is not part of this pinned candidate set.
    if (repository_ && !mandatory_blocks.empty()) {
        std::vector<uint8_t> available(n, 0);
        for (auto id : order) available[id] = 1;
        for (auto id : mandatory_blocks)
            if (!available.at(id))
                throw std::runtime_error("KVMem mandatory block is outside the pressure snapshot");
    }

    const uint32_t budget = prefill_budget_blocks();
    if (budget == 0 || order.size() <= budget) {
        selected.reserve(order.size());
        for (uint32_t id : order) if (is_live(id)) selected.push_back(id);
        return selected;
    }

    std::vector<uint8_t> kept(n, 0);
    uint32_t kept_count = 0;
    auto keep = [&](uint32_t id) {
        if (id < n && is_live(id) && !kept[id]) {
            kept[id] = 1;
            ++kept_count;
        }
    };
    for (uint32_t id : order) if (sink_block(id) && kept_count < budget) keep(id);
    for (uint32_t id : mandatory_blocks) keep(id);
    if (kept_count > budget) {
        throw std::runtime_error(
            "KVMem mandatory prefill selection plus sink blocks exceeds "
            "the configured prefill budget");
    }
    for (auto it = order.rbegin(); it != order.rend() && kept_count < budget; ++it) {
        keep(*it);
    }
    selected.reserve(kept_count);
    for (uint32_t id : order) {
        if (kept[id]) selected.push_back(id);
    }
    return selected;
}

std::vector<uint32_t> KvMemStore::pick_topk_blocks() const {
    return pick_topk_blocks({});
}

std::vector<uint32_t> KvMemStore::pick_topk_blocks(
        const std::vector<uint32_t> &mandatory_blocks) const {
    const uint32_t n = block_count();
    const auto order = chronological_ids();
    std::vector<uint32_t> selected;
    if (n == 0) return selected;
    gc_validate_mandatory(mandatory_blocks);

    const uint32_t budget = budget_blocks();
    if (budget == 0 || order.size() <= budget) {
        // Everything fits: select all in order.
        selected.reserve(n);
        for (uint32_t id : order) if (is_live(id)) selected.push_back(id);
        return selected;
    }

    // Always-keep windows: first `sink_blocks` and last `recent_blocks`.
    // Zero is literal: do not reserve any suffix blocks.  Earlier versions used
    // zero as an implicit "auto = budget/4", which made a 200K-token budget
    // silently pin 50K tokens and was both surprising and hard to control.
    const uint32_t recent = std::min<uint32_t>(cfg_.recent_blocks, order.size());

    std::vector<bool> kept(n, false);
    uint32_t kept_count = 0;
    auto keep = [&](uint32_t id) {
        if (id < n && is_live(id) && !kept[id]) { kept[id] = true; ++kept_count; }
    };
    for (uint32_t id : order) if (sink_block(id) && kept_count < budget) keep(id);
    for (uint32_t id : mandatory_blocks) keep(id);
    if (kept_count > budget) {
        throw std::runtime_error(
            "KVMem mandatory selection plus sink blocks exceeds the "
            "configured selection budget");
    }
    for (uint32_t i = 0; i < recent && kept_count < budget; ++i) {
        keep(order[order.size() - 1 - i]);
    }

    auto take_top = [&](uint32_t quota, auto score_fn) {
        if (kept_count >= budget || quota == 0) return;
        std::vector<uint32_t> candidates;
        candidates.reserve(n - kept_count);
        for (uint32_t i = 0; i < n; ++i) {
            if (!kept[i] && is_live(i)) candidates.push_back(i);
        }
        if (candidates.empty()) return;
        const uint32_t need = std::min<uint32_t>(
            std::min<uint32_t>(quota, budget - kept_count),
            static_cast<uint32_t>(candidates.size()));
        auto better = [&](uint32_t a, uint32_t b) {
            const double sa = score_fn(blocks_[a]);
            const double sb = score_fn(blocks_[b]);
            if (sa != sb) return sa > sb;
            return earlier(b, a);
        };
        if (need < candidates.size()) {
            std::nth_element(candidates.begin(), candidates.begin() + need,
                             candidates.end(), better);
            candidates.resize(need);
        }
        for (uint32_t id : candidates) keep(id);
    };

    if (cfg_.select_policy == KvMemSelectPolicy::Quota && kept_count < budget) {
        uint32_t remaining = budget - kept_count;
        uint32_t retrieval_quota = cfg_.retrieval_blocks;
        uint32_t profile_quota = cfg_.profile_blocks;
        if (retrieval_quota == 0 && profile_quota == 0) {
            retrieval_quota = (remaining * 2) / 3;
            profile_quota = remaining - retrieval_quota;
        } else if (retrieval_quota == 0) {
            retrieval_quota = remaining > profile_quota ? remaining - profile_quota : 0;
        } else if (profile_quota == 0) {
            profile_quota = remaining > retrieval_quota ? remaining - retrieval_quota : 0;
        }

        take_top(retrieval_quota, [](const KvMemBlock &b) {
            return b.retrieval_score;
        });
        take_top(profile_quota, [](const KvMemBlock &b) {
            return b.profile_score;
        });
        // Fill any leftover quota with the configured method's combined score
        // so rounding, overlap, or zero explicit quotas still use the budget.
        if (kept_count < budget) {
            take_top(budget - kept_count, [](const KvMemBlock &b) {
                return b.attn_score;
            });
        }
        selected.reserve(kept_count);
        for (uint32_t i : order) {
            if (kept[i]) selected.push_back(i);
        }
        return selected;
    }

    // Fill the rest with the highest cumulative-attention middle blocks.
    if (kept_count < budget) {
        std::vector<uint32_t> candidates;
        candidates.reserve(n - kept_count);
        for (uint32_t i = 0; i < n; ++i) {
            if (!kept[i] && is_live(i)) candidates.push_back(i);
        }
        const uint32_t need = budget - kept_count;
        // Partial sort by attn_score desc; tie-break by recency (higher id) so
        // selection is deterministic.
        auto better = [&](uint32_t a, uint32_t b) {
            if (blocks_[a].attn_score != blocks_[b].attn_score) {
                return blocks_[a].attn_score > blocks_[b].attn_score;
            }
            return earlier(b, a);
        };
        if (need < candidates.size()) {
            std::nth_element(candidates.begin(), candidates.begin() + need,
                             candidates.end(), better);
            candidates.resize(need);
        }
        for (uint32_t id : candidates) keep(id);
    }

    selected.reserve(kept_count);
    for (uint32_t i : order) {
        if (kept[i]) selected.push_back(i);
    }
    return selected;
}

std::vector<uint32_t> KvMemStore::pick_semantic_groups(
        const std::vector<std::pair<uint32_t, uint32_t>> &groups,
        const std::vector<double> &group_scores,
        const std::vector<uint32_t> &mandatory_blocks) const {
    if (repository_) {
        std::vector<std::pair<uint64_t, uint64_t>> spans(groups.begin(), groups.end());
        return pick_repository_semantic_groups(spans, group_scores, mandatory_blocks);
    }
    const uint32_t n = block_count();
    std::vector<uint32_t> selected;
    if (n == 0) return selected;
    gc_validate_mandatory(mandatory_blocks);
    if (groups.size() != group_scores.size()) {
        throw std::runtime_error(
            "KVMem semantic group spans/scores size mismatch");
    }

    const uint32_t budget = budget_blocks();
    if (budget == 0 || n <= budget) {
        selected.reserve(n);
        for (uint32_t id = 0; id < n; ++id) if (is_live(id)) selected.push_back(id);
        return selected;
    }

    std::vector<uint8_t> kept(n, 0);
    uint32_t kept_count = 0;
    auto keep = [&](uint32_t id) {
        if (id < n && is_live(id) && !kept[id]) {
            kept[id] = 1;
            ++kept_count;
        }
    };

    const uint32_t sink = std::min(cfg_.sink_blocks, n);
    const uint32_t recent = std::min(cfg_.recent_blocks, n);
    for (uint32_t id = 0; id < sink && kept_count < budget; ++id) keep(id);
    for (uint32_t id : mandatory_blocks) keep(id);
    if (kept_count > budget) {
        throw std::runtime_error(
            "KVMem mandatory semantic selection plus sink blocks exceeds the "
            "configured selection budget");
    }
    for (uint32_t i = 0; i < recent && kept_count < budget; ++i) {
        keep(n - 1 - i);
    }

    struct Candidate {
        uint32_t group = 0;
        uint32_t first = 0;
        uint32_t last = 0;
        double score = 0.0;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(groups.size());
    const uint32_t bt = std::max<uint32_t>(1, cfg_.block_tokens);
    const uint64_t covered_tokens =
        static_cast<uint64_t>(blocks_.back().orig_pos_start) +
        blocks_.back().n_tokens;
    for (uint32_t g = 0; g < groups.size(); ++g) {
        const uint32_t begin = groups[g].first;
        const uint32_t end = groups[g].second;
        if (end <= begin || begin >= covered_tokens) continue;
        const uint32_t capped_end = static_cast<uint32_t>(
            std::min<uint64_t>(end, covered_tokens));
        const uint32_t first = begin / bt;
        const uint32_t last = (capped_end - 1) / bt;
        if (first >= n) continue;
        candidates.push_back(Candidate{
            g, first, std::min(last, n - 1), group_scores[g]});
    }

    // GPU has already reduced fine-grained sub-block scores to one score per
    // logical group. Sorting O(groups log groups) is negligible for normal
    // conversation histories. Recency is the deterministic tie-break.
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &a, const Candidate &b) {
                  if (a.score != b.score) return a.score > b.score;
                  return a.group > b.group;
              });

    uint32_t selected_groups = 0;
    uint32_t skipped_for_budget = 0;
    for (const Candidate &candidate : candidates) {
        bool live_group = true;
        for (uint32_t id = candidate.first; id <= candidate.last; ++id) live_group = live_group && is_live(id);
        if (!live_group) continue;
        uint32_t new_blocks = 0;
        for (uint32_t id = candidate.first; id <= candidate.last; ++id) {
            new_blocks += kept[id] ? 0u : 1u;
        }
        if (new_blocks > budget - kept_count) {
            ++skipped_for_budget;
            continue;
        }
        for (uint32_t id = candidate.first; id <= candidate.last; ++id) {
            keep(id);
        }
        ++selected_groups;
        if (kept_count == budget) break;
    }

    selected.reserve(kept_count);
    for (uint32_t id = 0; id < n; ++id) {
        if (kept[id]) selected.push_back(id);
    }
    if (std::getenv("QW3_KVMEM_TRACE")) {
        std::fprintf(
            stderr,
            "[bs-semantic-select] mode=%s reduce=%s alpha=%.3f "
            "groups=%zu candidates=%zu selected_groups=%u "
            "selected_blocks=%u budget_blocks=%u skipped_for_budget=%u "
            "unused_blocks=%u\n",
            cfg_.semantic_expansion == KvMemSemanticExpansion::Message
                ? "message" : "round",
            cfg_.group_score_reduce ==
                    KvMemGroupScoreReduce::LengthNormalizedMass
                ? "length-normalized-mass" : "max",
            cfg_.group_length_norm_alpha,
            groups.size(), candidates.size(), selected_groups, kept_count,
            budget, skipped_for_budget, budget - kept_count);
    }
    return selected;
}

std::vector<uint32_t> KvMemStore::pick_repository_semantic_groups(
        const std::vector<std::pair<uint64_t, uint64_t>> &groups,
        const std::vector<double> &scores, const std::vector<uint32_t> &mandatory) const {
    if (!repository_ || groups.size() != scores.size())
        throw std::invalid_argument("invalid bounded KVMem group scores");
    gc_validate_mandatory(mandatory);
    const auto order = chronological_ids();
    const uint32_t budget = budget_blocks();
    if (!budget || order.size() <= budget) return order;
    std::vector<uint8_t> kept(block_count(), 0);
    uint32_t count = 0;
    auto keep = [&](uint32_t id) { if (!kept[id]) { kept[id] = 1; ++count; } };
    for (uint32_t id : order) if (sink_block(id) && count < budget) keep(id);
    for (uint32_t id : mandatory) keep(id);
    if (count > budget) throw std::runtime_error("KVMem mandatory semantic group union exceeds budget");
    for (size_t i = 0; i < order.size() && i < cfg_.recent_blocks && count < budget; ++i)
        keep(order[order.size() - 1 - i]);
    std::vector<size_t> ranked;
    ranked.reserve(groups.size());
    for (size_t i = 0; i < groups.size(); ++i) ranked.push_back(i);
    std::sort(ranked.begin(), ranked.end(), [&](size_t a, size_t b) {
        if (scores[a] != scores[b]) return scores[a] > scores[b];
        return a > b;
    });
    for (auto g : ranked) {
        const auto span = groups[g];
        if (span.second <= span.first || span.first >= total_tokens_) continue;
        std::vector<uint32_t> ids;
        try {
            ids = history_span_blocks(span.first, std::min(span.second, total_tokens_), true);
        } catch (const std::runtime_error &) {
            // A group with deleted history cannot silently become a partial
            // group by substituting a new generation at the same slot.
            continue;
        }
        uint32_t needed = 0;
        for (auto id : ids) needed += !kept[id];
        if (needed > budget - count) continue;
        for (auto id : ids) keep(id);
        if (count == budget) break;
    }
    std::vector<uint32_t> result;
    for (auto id : order) if (kept[id]) result.push_back(id);
    return result;
}

KvMemPlan KvMemStore::set_selection(std::vector<uint32_t> selected_ids,
                                    bool force_raw_refresh) {
    if (repository_) throw std::logic_error("bounded KVMem selection requires block generations");
    return set_selection_impl(std::move(selected_ids), force_raw_refresh);
}

KvMemPlan KvMemStore::set_repository_selection(
        const std::vector<KvMemBlockHandle> &handles, bool force_raw_refresh) {
    require_no_repository_append();
    if (!repository_) throw std::logic_error("KVMem repository selection requires bounded mode");
    std::vector<KvMemRepository::Lease> next;
    next.reserve(handles.size());
    std::vector<uint32_t> ids;
    ids.reserve(handles.size());
    for (const auto h : handles) {
        if (h != repository_handle(h.slot) || !h.generation)
            throw std::runtime_error("KVMem selection contains a stale generation");
        auto l = repository_->try_pin(h);
        if (!l) throw std::runtime_error("KVMem selection contains a retired block");
        ids.push_back(h.slot); next.push_back(std::move(l));
    }
    auto old = std::make_shared<std::vector<KvMemRepository::Lease>>(repository_checkpoint_leases());
    auto plan = set_selection_impl(std::move(ids), force_raw_refresh);
    plan.repository_previous_window = std::move(old);
    for (const auto &b : blocks_) {
        if (!b.in_working_set && b.tier != KvTier::GPU)
            repository_resident_leases_[b.block_id].reset();
    }
    for (auto &l : next) repository_resident_leases_[l.handle().slot] = std::move(l);
    if (repository_gc_) {
        for (const auto &b : blocks_) repository_sync_gc(b.block_id, b.in_working_set);
        if (repository_scores_pending_) {
            auto event = std::make_shared<KvMemRepositoryGc::Scores>();
            event->sequence = ++repository_score_sequence_;
            event->step = repository_gc_->step();
            event->entries.reserve(repository_score_handles_.size());
            for (const auto h : repository_score_handles_)
                if (h == repository_handle(h.slot) && is_live(h.slot) &&
                    (repository_gc_flags_[h.slot] & KvMemRepositoryGc::Indexed)) {
                    const auto &b = blocks_[h.slot];
                    event->entries.push_back({h, b.retrieval_score, b.in_working_set});
                }
            repository_gc_->submit_scores(std::move(event));
            repository_scores_pending_ = false;
        }
    }
    return plan;
}

KvMemPlan KvMemStore::set_selection_impl(std::vector<uint32_t> selected_ids,
                                       bool force_raw_refresh) {
    // Sort + dedupe so window order is deterministic (ascending block_id =
    // original chronological order: sink first ... recent last).
    // Drop out-of-range IDs defensively (external selector could be stale).
    selected_ids.erase(
        std::remove_if(selected_ids.begin(), selected_ids.end(),
                       [&](uint32_t id) { return id >= block_count(); }),
        selected_ids.end());
    std::sort(selected_ids.begin(), selected_ids.end(), [&](uint32_t a, uint32_t b) { return earlier(a, b); });
    selected_ids.erase(std::unique(selected_ids.begin(), selected_ids.end()), selected_ids.end());

    KvMemPlan plan;
    if (gc_) {
        gc_validate_mandatory(selected_ids);
        auto next = gc_->pin(selected_ids);
        plan.gc_previous_window = std::move(gc_working_lease_);
        gc_working_lease_ = std::move(next);
        gc_->set_active(gc_working_lease_);
    }

    std::vector<bool> now_selected(block_count(), false);
    for (uint32_t id : selected_ids) now_selected[id] = true;
    std::vector<bool> was_in_working_set(block_count(), false);
    for (const KvMemBlock &b : blocks_) {
        if (repository_ && !is_live(b.block_id)) continue;
        was_in_working_set[b.block_id] = b.in_working_set;
        if (b.in_working_set && now_selected[b.block_id]) {
            ++plan.selection_overlap_blocks;
        }
    }
    const bool force_full_reload = !cfg_.optimize_stage_in;

    // Stage-out: any GPU-resident block that is not selected. On the first
    // post-prefill selection, many cold blocks were never in the prior working
    // set, but they still occupy GPU KV pages and must be eligible for offload.
    // Their cache pages keep whatever bake they currently hold (baked_pos
    // unchanged); a future re-selection will de-rotate from there.
    for (auto &b : blocks_) {
        if (repository_ && !is_live(b.block_id)) continue;
        if (repository_ && !now_selected[b.block_id]) b.in_working_set = false;
        if (b.tier == KvTier::GPU &&
            (!now_selected[b.block_id] || force_full_reload)) {
            plan.stage_out.push_back(b.block_id);
            b.in_working_set = false;
        }
    }

    // Pack selected blocks contiguously into the window in ascending order.
    uint32_t window_pos = 0;
    for (uint32_t id : selected_ids) {
        KvMemBlock &b = blocks_[id];
        const bool naturally_reusable =
            was_in_working_set[id] && b.tier == KvTier::GPU;
        if (naturally_reusable) {
            if (b.baked_pos == static_cast<int64_t>(window_pos)) {
                ++plan.retained_position_stable;
            } else {
                ++plan.retained_position_moved;
            }
            if (!force_full_reload) ++plan.gpu_reused_blocks;
        }
        const bool cold = b.tier != KvTier::GPU || force_full_reload;
        if (!b.in_working_set) plan.stage_in.push_back(id);

        KvMemRemap rm;
        rm.block_id = id;
        rm.n_tokens = b.n_tokens;
        rm.from_base = static_cast<int32_t>(b.baked_pos);   // de-rotate source
        rm.to_base = static_cast<int32_t>(window_pos);      // new window slot
        rm.working_k_resident = !cold;
        const bool same_position =
            b.baked_pos == static_cast<int64_t>(window_pos);
        // A cold immutable block has no valid rotated working K on GPU: the
        // tier record intentionally stores V but uses the position-free raw-K
        // mirror as K authority. Even when its compact position is unchanged,
        // it must be rematerialized after stage-in. Position equality alone is
        // therefore insufficient to skip assembly.
        rm.skip = same_position &&
                  (!cfg_.immutable_source_k || rm.working_k_resident) &&
                  !force_raw_refresh;
        if (cfg_.immutable_source_k && (cold || !same_position)) {
            const uint64_t delta = static_cast<uint64_t>(
                b.baked_pos > static_cast<int64_t>(window_pos)
                    ? b.baked_pos - static_cast<int64_t>(window_pos)
                    : static_cast<int64_t>(window_pos) - b.baked_pos);
            const bool remap_limit =
                cfg_.immutable_refresh_remaps > 0 &&
                b.remap_count >= cfg_.immutable_refresh_remaps;
            const bool delta_limit =
                cfg_.immutable_refresh_abs_delta_tokens > 0 &&
                (b.remap_abs_delta >=
                     cfg_.immutable_refresh_abs_delta_tokens ||
                 delta >= cfg_.immutable_refresh_abs_delta_tokens -
                              std::min<uint64_t>(
                                  b.remap_abs_delta,
                                  cfg_.immutable_refresh_abs_delta_tokens));
            const bool baked_out_of_range =
                cfg_.immutable_max_baked_position > 0 &&
                (b.baked_pos < 0 ||
                 static_cast<uint64_t>(b.baked_pos) + b.n_tokens >
                     cfg_.immutable_max_baked_position);
            rm.raw_refresh =
                cold || remap_limit || delta_limit || baked_out_of_range;
        }
        if (cfg_.immutable_source_k && force_raw_refresh) {
            rm.raw_refresh = true;
        }
        if (rm.skip && rm.raw_refresh) {
            throw std::logic_error(
                "KVMem remap cannot both skip K assembly and refresh raw K");
        }
        plan.remaps.push_back(rm);

        b.in_working_set = true;
        if (!rm.skip) {
            const uint64_t delta = static_cast<uint64_t>(
                b.baked_pos > static_cast<int64_t>(window_pos)
                    ? b.baked_pos - static_cast<int64_t>(window_pos)
                    : static_cast<int64_t>(window_pos) - b.baked_pos);
            if (rm.raw_refresh) {
                b.remap_count = 0;
                b.remap_abs_delta = 0;
            } else {
                ++b.remap_count;
                b.remap_abs_delta += delta;
            }
        }
        b.baked_pos = static_cast<int64_t>(window_pos);
        window_pos += b.n_tokens;
    }
    plan.total_window_tokens = window_pos;
    return plan;
}

void KvMemStore::clear_working_set() {
    require_no_repository_append();
    for (auto &b : blocks_) {
        b.in_working_set = false;
        if (repository_ && b.tier != KvTier::GPU)
            repository_resident_leases_[b.block_id].reset();
        repository_sync_gc(b.block_id);
    }
}

void KvMemStore::gc_sync_block(uint32_t id) {
    const auto &b = blocks_[id];
    const uint32_t mask = KvMemGc::Sealed | KvMemGc::Gpu | KvMemGc::Io | KvMemGc::ValueBacked | KvMemGc::CpuCached;
    const uint32_t flags = (b.n_tokens == cfg_.block_tokens ? KvMemGc::Sealed : 0u) |
        (b.tier == KvTier::GPU ? KvMemGc::Gpu : 0u) | (b.in_flight ? KvMemGc::Io : 0u) |
        (b.ssd_clean && b.nvme_slot >= 0 ? KvMemGc::ValueBacked : 0u) |
        (b.cpu_slot >= 0 ? KvMemGc::CpuCached : 0u);
    gc_->update_flags(id, mask, flags);
}
void KvMemStore::gc_validate_mandatory(const std::vector<uint32_t> &ids) const {
    if (!gc_ && !repository_) return;
    for (uint32_t id : ids) if ((repository_ || id < block_count()) && !is_live(id))
        throw std::runtime_error("KVMem GC: mandatory/checkpoint block was deleted");
}
void KvMemStore::gc_begin_step() {
    if (repository_gc_) {
        repository_gc_->begin_step();
        for (const auto &b : blocks_)
            if (b.in_working_set || b.tier == KvTier::GPU) repository_sync_gc(b.block_id, true);
    }
    if (!gc_) return;
    gc_->begin_step();
    gc_view_ = gc_->read_view();
}
uint32_t KvMemStore::gc_missing_required_prefix_block(uint32_t begin, uint32_t end, uint32_t prefix_end) const {
    if (!gc_) return UINT32_MAX;
    end = std::min(end, prefix_end);
    if (end <= begin) return UINT32_MAX;
    const uint32_t last = std::min(block_count(), (end - 1) / cfg_.block_tokens + 1);
    for (uint32_t id = begin / cfg_.block_tokens; id < last; ++id) if (!is_live(id)) return id;
    return UINT32_MAX;
}
std::shared_ptr<KvMemGc::Lease> KvMemStore::gc_checkpoint_lease() {
    if (!gc_) return {};
    // Called at durable request checkpoints, never per decode token. Include
    // the newly appended GPU tail as well as the selected historical window.
    std::vector<uint32_t> ids;
    for (const auto &b : blocks_) if (b.in_working_set || b.tier == KvTier::GPU) ids.push_back(b.block_id);
    return gc_->pin(ids);
}

} // namespace qw3
