#include "qwen_executor.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>

namespace qw3 {
namespace {
void mismatch_require(DeviceStatus s) {
    if (!s.ok) throw std::runtime_error(s.message ? s.message : "mismatch capture failed");
}
double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
}
void QwenExecutor::kvmem_mismatch_configure(const KvMemStoreConfig &cfg) {
    cfg.mismatch.validate();
    mismatch_config_=cfg.mismatch;
    if (!cfg.mismatch.enabled()) { mismatch_observer_.reset(); return; }
    if (cfg.select_method!=KvMemMethod::Retrieval || cfg.index_placement!=KvMemIndexPlacement::CPU ||
        cfg.retrieval_method!=KvMemRetrievalMethod::MeanK || cfg.n_subblocks!=1 ||
        cfg.prototype_mode!=KvMemPrototypeMode::Contiguous || cfg.repository_blocks ||
        cfg.gc.max_live_blocks)
        throw std::invalid_argument("CPU mismatch observation requires an unbounded CPU mean-k index (no repository/GC)");
    if (cfg.mismatch.query_mode=="block-mean" && !cfg.immutable_source_k)
        throw std::invalid_argument("block-mean mismatch observation requires --kvmem-immutable-k for the complete history");
    std::vector<int32_t> standard;
    for (uint32_t l=0;l<weights_.n_layers();++l)
        if (model_.config().is_standard_attention_layer(l)) standard.push_back(l);
    mismatch_layers_=detail::resolve_attention_layers(cfg.mismatch.layers,standard);
    mismatch_observer_=std::make_unique<detail::MismatchObserver>(cfg.mismatch);
    mismatch_parser_.scope=detail::ActionAttentionParser::Scope::AllOutput;
    mismatch_parser_.interval=cfg.mismatch.interval;
    kvmem_mismatch_reset();
    mismatch_q_device_.reset(); mismatch_q_host_.reset();
    std::fprintf(stderr,"[kvmem-mismatch] observe_only=1 device=cpu interval=%u samples=%u layers=%s log=%s\n",
        cfg.mismatch.interval,cfg.mismatch.samples,cfg.mismatch.layers.c_str(),cfg.mismatch.log_path.c_str());
}

void QwenExecutor::kvmem_mismatch_reset() {
    if (!mismatch_observer_) return;
    // Only drains the tiny Q DMA, never the CPU scoring worker.
    if (mismatch_copy_done_) mismatch_require(backend_.wait_kv_transfer_fence(mismatch_copy_done_));
    ++mismatch_epoch_;
    mismatch_index_update_ms_=0; mismatch_index_copy_bytes_=0;
    mismatch_index_.clear(); mismatch_rows_.clear(); mismatch_selected_.clear();
    mismatch_mandatory_.clear();
    mismatch_prefill_last_=mismatch_committed_=0;
    mismatch_running_=false;
    mismatch_parser_.committed={}; mismatch_parser_.candidate.clear();
    mismatch_parser_.mask=0; mismatch_parser_.accepted=0;
    mismatch_query_mean_.reset();
}
void QwenExecutor::kvmem_mismatch_begin(int32_t think_open,int32_t think_close,bool thinking) {
    if (!mismatch_observer_) return;
    if (think_open<0 || think_close<0) throw std::invalid_argument("mismatch observation requires think special tokens");
    mismatch_running_=true;
    mismatch_parser_.think_open=think_open; mismatch_parser_.think_close=think_close;
    mismatch_parser_.committed.thinking=thinking;
    mismatch_parser_.committed.action=false;
    mismatch_parser_.candidate.clear(); mismatch_parser_.mask=0; mismatch_parser_.accepted=0;
}

void QwenExecutor::kvmem_mismatch_prepare(const uint32_t *tokens,uint32_t count,uint64_t base,bool prefill) {
    if (!mismatch_observer_) return;
    if (mismatch_copy_done_) mismatch_require(backend_.wait_kv_transfer_fence(mismatch_copy_done_));
    mismatch_rows_.clear(); mismatch_captured_layers_=0; mismatch_capture_ms_=0;
    mismatch_count_=0; mismatch_accepted_=0;
    if (kvmem_query_probe_guard_active() || kvmem_query_replay_active_ ||
        kvmem_provisional_prefill_ || kvmem_blend_active_ || !count) return;
    if (base<mismatch_committed_) kvmem_mismatch_reset();
    mismatch_base_=base; mismatch_count_=count;
    mismatch_prefill_=prefill && !mismatch_running_;
    if (!mismatch_prefill_ && !mismatch_running_) { mismatch_count_=0; return; }
    const bool block_mean=mismatch_config_.query_mode=="block-mean";
    bool due=block_mean;
    if (!block_mean && mismatch_prefill_) due=!mismatch_prefill_last_ || base+count-mismatch_prefill_last_>=mismatch_config_.interval;
    else if (!block_mean) { mismatch_parser_.prepare(tokens,count); due=mismatch_parser_.mask!=0; }
    if (!due) return;
    if (!block_mean && mismatch_observer_->busy()) { ++mismatch_skipped_busy_; return; }
    if (block_mean) {
        mismatch_rows_.resize(count);std::iota(mismatch_rows_.begin(),mismatch_rows_.end(),0);
    } else if (mismatch_prefill_) {
        const uint32_t samples=std::min(count,mismatch_config_.samples);
        for (uint32_t i=0;i<samples;++i) mismatch_rows_.push_back(uint64_t(i+1)*count/samples-1);
    } else {
        for (uint32_t i=0;i<std::min(64u,count) && mismatch_rows_.size()<mismatch_config_.samples;++i)
            if ((mismatch_parser_.mask>>i)&1) mismatch_rows_.push_back(i);
    }
    const auto &cfg=model_.config();
    const uint64_t elems=uint64_t(mismatch_layers_.size())*(block_mean?count:mismatch_config_.samples)*cfg.n_heads*cfg.head_dim;
    if (!mismatch_q_device_ || mismatch_q_device_->count<elems) {
        mismatch_q_device_=backend_.tensor_f16(elems,"mismatch Q");
        mismatch_q_host_=backend_.host_buffer(elems*sizeof(uint16_t),"mismatch Q host");
    }
}

void QwenExecutor::kvmem_mismatch_capture(uint32_t layer,const DeviceTensor &q,
        uint32_t q_stride,uint32_t rope_base,const DeviceTensor *mrope,uint32_t mrope_stride) {
    if (mismatch_rows_.empty()) return;
    auto it=std::lower_bound(mismatch_layers_.begin(),mismatch_layers_.end(),int32_t(layer));
    if (it==mismatch_layers_.end() || *it!=int32_t(layer)) return;
    const auto start=std::chrono::steady_clock::now();
    const size_t l=it-mismatch_layers_.begin();
    const auto &cfg=model_.config();
    const uint64_t row_elems=uint64_t(cfg.n_heads)*cfg.head_dim;
    if (l==0) {
        mismatch_selection_=kvmem_selection_generation_;
        mismatch_selected_.clear();
        mismatch_mandatory_.clear();
        uint32_t prefix=0;
        if (kvmem_active_) {
            const auto &store_cfg=block_store_->config();
            const uint32_t nb=block_store_->block_count();
            const uint32_t recent_begin=nb-std::min(nb,store_cfg.recent_blocks);
            const uint64_t history_end=block_store_->history_end();
            const auto overlaps=[&](uint32_t id,const auto &spans) {
                const uint64_t lo=uint64_t(id)*store_cfg.block_tokens;
                const uint64_t hi=lo+store_cfg.block_tokens;
                return std::any_of(spans.begin(),spans.end(),[&](const auto &span) {
                    return span.first<hi && span.second>lo;
                });
            };
            for (uint32_t w=0;w<bs_window_blocks_;++w) {
                const uint32_t id=bs_window_block_ids_[w];
                const uint32_t tokens=bs_blk_tokens_host_[w];
                mismatch_selected_.emplace_back(id,tokens);
                // This defines a reference mask, not a new retention policy.
                // Keep the already-resident sink/recent/request-pin bands and
                // unfinished prefix fixed, randomizing other token slots only.
                // In the sink1K/recent0/no-pins experiment this is just sink
                // plus the partial append boundary. No selector is invoked.
                const bool fixed=id<store_cfg.sink_blocks || id>=recent_begin ||
                    id>=kvmem_qc_pin_from_block_ ||
                    (history_end%store_cfg.block_tokens && nb && id==nb-1) ||
                    overlaps(id,kvmem_mandatory_token_spans_) ||
                    overlaps(id,kvmem_oracle_token_spans_);
                if (fixed) mismatch_mandatory_.emplace_back(id,tokens);
                prefix=bs_win_base_host_[w]+bs_blk_tokens_host_[w];
            }
        }
        const uint64_t physical=kvmem_active_?window_query_pos_:mismatch_base_;
        if (physical<prefix || physical-prefix>mismatch_base_)
            throw std::logic_error("mismatch probe window origin is invalid");
        mismatch_tail_begin_=mismatch_base_-(physical-prefix);
    }
    if (mismatch_config_.query_mode=="block-mean") {
        const uint64_t dest=l*mismatch_rows_.size()*row_elems;
        if (mrope) mismatch_require(backend_.derope_query_multi_mrope_device(
            *mismatch_q_device_,q,0,dest,q_stride,2*cfg.head_dim,mismatch_count_,
            cfg.n_heads,cfg.head_dim,cfg.rope_dim,*mrope,0,mrope_stride,cfg.rope_theta));
        else mismatch_require(backend_.derope_query_multi_device(
            *mismatch_q_device_,q,0,dest,q_stride,2*cfg.head_dim,mismatch_count_,
            cfg.n_heads,cfg.head_dim,cfg.rope_dim,int32_t(rope_base),cfg.rope_theta));
    } else for (size_t r=0;r<mismatch_rows_.size();++r) {
        const uint32_t row=mismatch_rows_[r];
        const uint64_t dest=(l*mismatch_rows_.size()+r)*row_elems;
        if (mrope) mismatch_require(backend_.derope_query_multi_mrope_device(
            *mismatch_q_device_,q,uint64_t(row)*q_stride,dest,q_stride,2*cfg.head_dim,1,
            cfg.n_heads,cfg.head_dim,cfg.rope_dim,*mrope,row,mrope_stride,cfg.rope_theta));
        else mismatch_require(backend_.derope_query_multi_device(
            *mismatch_q_device_,q,uint64_t(row)*q_stride,dest,q_stride,2*cfg.head_dim,1,
            cfg.n_heads,cfg.head_dim,cfg.rope_dim,int32_t(rope_base+row),cfg.rope_theta));
    }
    ++mismatch_captured_layers_;
    if (mismatch_captured_layers_==mismatch_layers_.size()) {
        mismatch_require(backend_.begin_kv_transfer_from_device());
        mismatch_require(backend_.copy_bytes_to_host_async(*mismatch_q_device_,mismatch_q_host_->data,0,
            mismatch_layers_.size()*mismatch_rows_.size()*row_elems*sizeof(uint16_t)));
        mismatch_require(backend_.record_kv_transfer_fence(mismatch_copy_done_));
    }
    mismatch_capture_ms_+=elapsed_ms(start);
}

void QwenExecutor::kvmem_mismatch_index_update(uint64_t first,uint32_t count) {
    if (!mismatch_observer_ || !count || g_kbar_multi_host_elem_size_!=2 || !block_store_) return;
    const auto update_start=std::chrono::steady_clock::now();
    const auto &cfg=model_.config();
    const uint64_t width=uint64_t(cfg.n_kv_heads)*cfg.head_dim;
    if (first+count>g_kbar_multi_host_layer_bytes_/(width*2)) return;
    if (mismatch_index_.size()<first+count) mismatch_index_.resize(first+count);
    for (uint64_t b=first;b<first+count;++b) {
        auto block=std::make_shared<detail::MismatchKeyBlock>();
        block->keys.resize(width*mismatch_layers_.size());
        for (size_t l=0;l<mismatch_layers_.size();++l) {
            const auto layer=mismatch_layers_[l];
            if (size_t(layer)>=std_layer_slot_.size() || std_layer_slot_[layer]<0)
                throw std::runtime_error("mismatch layer missing from the Mean-K index; remove QW3_KVMEM_QC_LAYERS cap");
            const auto *src=kvmem_mean_index_host_layer_data(std_layer_slot_[layer]);
            if (!src) throw std::runtime_error("missing mismatch CPU index layer");
            std::memcpy(block->keys.data()+l*width,src+b*width*2,width*2);
        }
        mismatch_index_[b]=std::move(block);
    }
    mismatch_index_update_ms_+=elapsed_ms(update_start);
    mismatch_index_copy_bytes_+=uint64_t(count)*width*mismatch_layers_.size()*2;
}

void QwenExecutor::kvmem_mismatch_commit(uint64_t target) {
    if (!mismatch_observer_ || !mismatch_count_ || target<=mismatch_base_ ||
        kvmem_query_probe_guard_active() || kvmem_query_replay_active_ ||
        kvmem_provisional_prefill_ || kvmem_blend_active_) return;
    const uint32_t accepted=std::min<uint64_t>(target-mismatch_base_,mismatch_count_);
    const bool block_mean=mismatch_config_.query_mode=="block-mean";
    uint64_t mask=~uint64_t(0);
    if (!mismatch_prefill_ && !block_mean) mask=mismatch_parser_.commit(accepted);
    mismatch_committed_=target;
    if (mismatch_rows_.empty() || mismatch_captured_layers_!=mismatch_layers_.size()) return;
    std::vector<size_t> rows;
    for (size_t r=0;r<mismatch_rows_.size();++r)
        if (mismatch_rows_[r]<accepted && (block_mean || mismatch_prefill_ || ((mask>>mismatch_rows_[r])&1))) rows.push_back(r);
    if (rows.empty()) return;
    const auto start=std::chrono::steady_clock::now();
    mismatch_require(backend_.wait_kv_transfer_fence(mismatch_copy_done_));
    const double capture_ms=mismatch_capture_ms_+elapsed_ms(start);
    const auto snapshot_start=std::chrono::steady_clock::now();
    const auto &cfg=model_.config();
    const uint32_t bt=block_store_->config().block_tokens;
    const uint64_t query_width=uint64_t(cfg.n_heads)*cfg.head_dim;
    const auto *q=static_cast<const uint16_t *>(mismatch_q_host_->data);
    std::vector<detail::MismatchMeanQuery> means;
    uint64_t endpoint=target;
    if (block_mean) {
        means=mismatch_query_mean_.append(q,mismatch_layers_.size(),mismatch_count_,accepted,
            query_width,mismatch_base_,bt,mismatch_accepted_);
        if (!mismatch_prefill_) mismatch_parser_.committed.tokens+=accepted-mismatch_accepted_;
        mismatch_accepted_=accepted;
        if (means.empty()) return;
        if (mismatch_observer_->busy()) {
            ++mismatch_skipped_busy_;mismatch_skipped_query_blocks_+=means.size();return;
        }
        endpoint=means.back().position+1;
    }
    kvmem_drain_mean_index_capture();
    const uint32_t official_blocks=std::min<uint64_t>(kvmem_qc_captured_tokens_,endpoint)/bt;
    const uint32_t blocks=block_mean ? endpoint/bt : official_blocks;
    // Prefix-cache restore may already contain completed canonical index rows.
    // Materialize missing immutable blocks once, not a full copy every probe.
    for (uint32_t b=0;b<official_blocks;) {
        if (b<mismatch_index_.size() && mismatch_index_[b]) { ++b; continue; }
        const uint32_t first=b++;
        while (b<official_blocks && (b>=mismatch_index_.size() || !mismatch_index_[b])) ++b;
        kvmem_mismatch_index_update(first,b-first);
    }
    if (block_mean) {
        if (mismatch_index_.size()<blocks) mismatch_index_.resize(blocks);
        const uint64_t width=uint64_t(cfg.n_kv_heads)*cfg.head_dim;
        const uint32_t bytes=kvmem_raw_k_row_bytes_/width;
        if ((bytes!=1 && bytes!=2 && bytes!=4) || bytes*width!=kvmem_raw_k_row_bytes_)
            throw std::logic_error("unsupported diagnostic raw K storage");
        for (uint32_t b=official_blocks;b<blocks;++b) if (!mismatch_index_[b]) {
            const uint64_t lo=uint64_t(b)*bt, hi=lo+bt;
            if (hi>kvmem_raw_k_valid_tokens_.size() ||
                !std::all_of(kvmem_raw_k_valid_tokens_.begin()+lo,kvmem_raw_k_valid_tokens_.begin()+hi,
                    [](uint8_t v){return v!=0;}))
                throw std::logic_error("missing accepted raw K for diagnostic history");
            auto block=std::make_shared<detail::MismatchKeyBlock>();
            block->raw_tokens=bt;block->raw_elem_bytes=bytes;
            block->raw.resize(mismatch_layers_.size()*bt*kvmem_raw_k_row_bytes_);
            for (size_t l=0;l<mismatch_layers_.size();++l)
                kvmem_read_raw_k(kvmem_raw_layer_slot_.at(mismatch_layers_[l]),lo,
                    block->raw.data()+l*bt*kvmem_raw_k_row_bytes_,bt,false);
            mismatch_index_[b]=std::move(block);
        }
    }
    detail::MismatchJob job;
    job.epoch=mismatch_epoch_; job.selection=mismatch_selection_; job.position=endpoint;
    job.progress=mismatch_prefill_?target:mismatch_parser_.committed.tokens;
    job.indexed_tokens=uint64_t(blocks)*bt; job.tail_begin=mismatch_tail_begin_;
    job.official_indexed_tokens=uint64_t(official_blocks)*bt;
    job.skipped_query_blocks=mismatch_skipped_query_blocks_;
    job.discarded_query_tokens=mismatch_query_mean_.discarded_tokens;
    job.skipped_busy=mismatch_skipped_busy_; job.block_tokens=bt;
    job.heads=cfg.n_heads; job.kv_heads=cfg.n_kv_heads; job.dim=cfg.head_dim;
    job.sink_blocks=block_store_->config().sink_blocks; job.prefill=mismatch_prefill_;
    job.layers=mismatch_layers_; job.capture_ms=capture_ms;
    if (mismatch_index_.size()<blocks) return; // No authoritative host index yet.
    job.blocks.assign(mismatch_index_.begin(),mismatch_index_.begin()+blocks);
    job.selected_tokens.assign(blocks,0); job.live.assign(blocks,1);
    job.mandatory_tokens.assign(blocks,0);
    for (const auto &[id,tokens]:mismatch_selected_) if (id<blocks) job.selected_tokens[id]=tokens;
    for (const auto &[id,tokens]:mismatch_mandatory_) if (id<blocks) job.mandatory_tokens[id]=tokens;
    const uint64_t width=uint64_t(cfg.n_heads)*cfg.head_dim;
    if (block_mean) {
        for (const auto &m:means) { job.query_positions.push_back(m.position);job.query_token_counts.push_back(m.tokens); }
        for (size_t l=0;l<mismatch_layers_.size();++l) for (const auto &m:means)
            job.mean_queries.insert(job.mean_queries.end(),m.values.begin()+l*width,m.values.begin()+(l+1)*width);
    } else {
        for (size_t r:rows) { job.query_positions.push_back(mismatch_base_+mismatch_rows_[r]);job.query_token_counts.push_back(1); }
        for (size_t l=0;l<mismatch_layers_.size();++l) for (size_t r:rows) {
            const auto *src=q+(l*mismatch_rows_.size()+r)*width;
            job.queries.insert(job.queries.end(),src,src+width);
        }
    }
    job.snapshot_ms=elapsed_ms(snapshot_start);
    job.index_update_ms_total=mismatch_index_update_ms_;
    job.index_copy_bytes_total=mismatch_index_copy_bytes_;
    if (!mismatch_observer_->submit(std::move(job))) {
        ++mismatch_skipped_busy_;mismatch_skipped_query_blocks_+=means.size();
    }
    if (mismatch_prefill_) mismatch_prefill_last_=target;
    // Prefill commits whole chunks; output commit's parser mask is idempotent.
    if (mismatch_prefill_) mismatch_rows_.clear();
}
}
