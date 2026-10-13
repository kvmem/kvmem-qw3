#include "qwen_executor.hpp"
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace qw3 {
struct KvPageTableTestAccess {
static void bench(const QwenNativeModel &model,const QwenWeights &weights,DeviceBackend &backend,const KvMemAttentionConfig &attention) {
    auto ck=[](bool ok,const char *s){if(!ok)throw std::runtime_error(s);};
    QwenExecutor e(model,weights,backend,65536);
    KvMemStoreConfig c;c.block_tokens=32;c.select_budget=c.prefill_budget=36864;
    c.gen_budget=16384;c.sink_blocks=41;c.recent_blocks=0;c.immutable_source_k=true;
    c.cpu_tier_bytes=8ull<<30;c.gpu_memory_ratio=.7;
    c.attention=attention;
    e.configure_kvmem(c);e.set_kvmem_enabled(true);e.set_prefill_chunk_override(1024);
    e.kvmem_set_query_span(40928,40960,40960);
    std::vector<uint32_t> history(40960);for(uint32_t i=0;i<history.size();++i)history[i]=200+i%200;
    ck(e.forward_n_tokens(history,false).ok,"benchmark prefill");e.kvmem_register_append(history.size());
    std::vector<uint32_t> ids(1152);for(uint32_t i=0;i<ids.size();++i)ids[i]=i;
    e.kvmem_set_selection(ids);auto boundary=e.snapshot_state();
    e.action_parser_.interval=16;
    // Hold the selected window and all teacher-forced tokens constant, avoiding
    // output length / acceptance changes as a confound in overhead measurement.
    for(unsigned width:{1u,8u}) for(unsigned trial=0;trial<6;++trial) for(unsigned order=0;order<2;++order) {
        bool enabled=(order+trial)%2;
        e.restore_state(boundary);e.kvmem_action_begin(100,101,102,103,false);
        e.action_running_=enabled;
        const unsigned steps=128/width;
        backend.synchronize();auto begin=std::chrono::steady_clock::now();
        for(unsigned step=0;step<steps;++step) {
            std::vector<uint32_t> t(width,200);
            if(step==0 && e.action_parser_.scope==detail::ActionAttentionParser::Scope::ToolCall)t[0]=100;
            std::vector<DeviceArgmax> verified;
            ck((width==1?e.forward_one_token(t[0],true):e.forward_n_tokens(t,true,&verified)).ok,"benchmark forward");
            e.kvmem_register_append(width);
        }
        e.kvmem_action_end();backend.synchronize();
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        std::printf("MODEL_BENCH window=36864 gen_budget=16384 width=%u trial=%u enabled=%d tokens=128 seconds=%.6f tps=%.3f scope=%s\n",width,trial,enabled,seconds,128/seconds,e.action_parser_.scope_name());
        std::fflush(stdout);
    }
    // Time the newly added host-only pressure policy separately from collection
    // and GPU remapping. Repeated inspection must not decay/commit history.
    for(unsigned trial=0;trial<6;++trial) for(unsigned order=0;order<2;++order) {
        bool enabled=(order+trial)%2;
        e.action_pressure_=enabled;
        const unsigned repeats=500;
        auto start=std::chrono::steady_clock::now();
        uint64_t selected=0;
        for(unsigned i=0;i<repeats;++i) {
            const auto scores=e.kvmem_action_pressure_scores();
            selected+=e.block_store_->pick_prefill_pressure_blocks({},scores).size();
        }
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        ck(selected==uint64_t(repeats)*1152,"pressure benchmark selection budget");
        std::printf("PRESSURE_BENCH trial=%u enabled=%d repeats=%u seconds=%.6f us_per_call=%.3f\n",
                    trial,enabled,repeats,seconds,seconds*1e6/repeats);
        std::fflush(stdout);
    }
}
static void bench_prefill(const QwenNativeModel &model,const QwenWeights &weights,DeviceBackend &backend,const KvMemAttentionConfig &attention) {
    auto ck=[](bool ok,const char *s){if(!ok)throw std::runtime_error(s);};
    QwenExecutor e(model,weights,backend,65536);
    KvMemStoreConfig c;c.block_tokens=32;c.select_budget=c.prefill_budget=36864;
    c.gen_budget=16384;c.sink_blocks=41;c.recent_blocks=0;c.immutable_source_k=true;
    c.cpu_tier_bytes=8ull<<30;c.gpu_memory_ratio=.94;
    c.attention=attention;
    e.configure_kvmem(c);e.set_kvmem_enabled(true);e.set_prefill_chunk_override(1024);
    const std::vector<uint32_t> prompt(36864,200);
    // Same full prompt, same chunks, no eviction: collection overhead only.
    // Warm up allocator/kernel shapes; reset and setup stay outside timing.
    for(unsigned trial=0;trial<5;++trial) for(unsigned order=0;order<2;++order) {
        const bool enabled=(trial+order)%2;
        e.reset_state();e.prefill_attention_enabled_=enabled;
        backend.synchronize();auto begin=std::chrono::steady_clock::now();
        ck(e.forward_n_tokens(prompt,false).ok,"prefill benchmark");e.kvmem_register_append(prompt.size());
        backend.synchronize();
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        ck(e.prefill_attention_samples_==(enabled?5:0),"prefill benchmark sample count");
        std::printf("PREFILL_BENCH trial=%u enabled=%d warmup=%d tokens=36864 chunk=1024 samples=%llu seconds=%.6f tps=%.3f\n",
                    trial,enabled,trial==0,static_cast<unsigned long long>(e.prefill_attention_samples_),seconds,prompt.size()/seconds);
        std::fflush(stdout);
    }
}
static void bench_sampling(const QwenNativeModel &model,const QwenWeights &weights,DeviceBackend &backend,KvMemAttentionConfig attention,bool prefill,bool dense=false) {
    auto ck=[](bool ok,const char *s){if(!ok)throw std::runtime_error(s);};
    QwenExecutor e(model,weights,backend,65536);
    KvMemStoreConfig c;c.block_tokens=32;c.select_budget=c.prefill_budget=36864;
    c.gen_budget=16384;c.sink_blocks=41;c.recent_blocks=0;c.immutable_source_k=true;
    c.cpu_tier_bytes=8ull<<30;c.gpu_memory_ratio=.94;
    attention.scope="non-thinking";attention.prefill=true;attention.prefill_stride=0;
    attention.prefill_interval=dense?1024:8192;attention.interval=16;c.attention=attention;
    e.configure_kvmem(c);e.set_kvmem_enabled(true);e.set_prefill_chunk_override(1024);
    std::vector<int32_t> standard;
    for(uint32_t i=0;i<weights.n_layers();++i)if(model.config().is_standard_attention_layer(i))standard.push_back(i);
    struct Mode { const char *name; const char *layers; uint32_t stride; bool enabled; };
    const std::vector<Mode> modes = dense ? std::vector<Mode>{
        {"off","last",0,false},{"spread3-256","spread3",256,true},{"spread3-128","spread3",128,true},
        {"spread3-64","spread3",64,true},{"spread3-32","spread3",32,true}
    } : prefill ? std::vector<Mode>{
        {"off","last",0,false},{"last-row","last",0,true},{"spread3-row","spread3",0,true},
        {"spread3-256","spread3",256,true},{"spread3-128","spread3",128,true},
        {"spread3-64","spread3",64,true},{"spread3-32","spread3",32,true},
        {"last-32","last",32,true}
    } : std::vector<Mode>{{"off","last",0,false},{"last","last",0,true},{"spread3","spread3",0,true},{"all","all",0,true}};
    const std::vector<uint32_t> prompt(36864,200);
    auto boundary=e.snapshot_state();
    if(!prefill) {
        e.prefill_attention_enabled_=false;
        e.kvmem_set_query_span(40928,40960,40960);
        std::vector<uint32_t> history(40960);for(uint32_t i=0;i<history.size();++i)history[i]=200+i%200;
        ck(e.forward_n_tokens(history,false).ok,"sampling benchmark prefill");e.kvmem_register_append(history.size());
        std::vector<uint32_t> ids(1152);for(uint32_t i=0;i<ids.size();++i)ids[i]=i;
        e.kvmem_set_selection(ids);boundary=e.snapshot_state();
    }
    for(unsigned width : prefill ? std::vector<unsigned>{1024} : std::vector<unsigned>{1,8})
    for(unsigned trial=0;trial<6;++trial) for(unsigned order=0;order<modes.size();++order) {
        const auto &mode=modes[(trial%2 ? modes.size()-1-order+trial : order+trial)%modes.size()];
        if(prefill)e.reset_state();else e.restore_state(boundary);
        e.action_layers_=detail::resolve_attention_layers(mode.layers,standard);
        e.prefill_attention_stride_=mode.stride;
        e.prefill_attention_enabled_=prefill && mode.enabled;
        if(!prefill){e.kvmem_action_begin(100,101,102,103,false);e.action_running_=mode.enabled;}
        backend.synchronize();const auto start=std::chrono::steady_clock::now();
        if(prefill) {
            ck(e.forward_n_tokens(prompt,false).ok,"sampling prefill benchmark");e.kvmem_register_append(prompt.size());
            const unsigned rows=mode.stride?1+(1024-1)/mode.stride:1;
            ck(e.prefill_attention_samples_==(mode.enabled?(dense?36:5)*rows:0),"sampling benchmark prefill count");
        } else {
            for(unsigned step=0;step<128/width;++step) {
                std::vector<uint32_t> t(width,200);std::vector<DeviceArgmax> verified;
                ck((width==1?e.forward_one_token(t[0],true):e.forward_n_tokens(t,true,&verified)).ok,"sampling output benchmark");
                e.kvmem_register_append(width);
            }
            e.kvmem_action_end();
            ck(e.action_round_samples_==(mode.enabled?8:0),"sampling benchmark output count");
        }
        backend.synchronize();const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::printf("SAMPLING_BENCH phase=%s mode=%s trial=%u warmup=%d width=%u layers=%zu stride=%u chunk_interval=%u tokens=%u samples=%llu seconds=%.6f tps=%.3f\n",
            prefill?"prefill":"output",mode.name,trial,trial==0,width,e.action_layers_.size(),mode.stride,e.prefill_attention_interval_,prefill?36864:128,
            static_cast<unsigned long long>(prefill?e.prefill_attention_samples_:e.action_round_samples_),seconds,(prefill?36864:128)/seconds);
        std::fflush(stdout);
    }
}
static void layer_equivalence(const QwenNativeModel &model,const QwenWeights &weights,DeviceBackend &backend,KvMemAttentionConfig attention) {
    auto ck=[](bool ok,const char *s){if(!ok)throw std::runtime_error(s);};
    attention.layers="spread3";attention.scope="non-thinking";attention.interval=1;attention.prefill=false;
    QwenExecutor e(model,weights,backend,4096);
    KvMemStoreConfig c;c.block_tokens=32;c.select_budget=c.prefill_budget=192;c.gen_budget=1024;
    c.sink_blocks=1;c.recent_blocks=0;c.immutable_source_k=true;c.cpu_tier_bytes=1ull<<30;c.gpu_memory_ratio=1;c.attention=attention;
    e.configure_kvmem(c);e.set_kvmem_enabled(true);
    ck(e.forward_n_tokens(std::vector<uint32_t>(512,200),false).ok,"equivalence prefill");e.kvmem_register_append(512);
    const auto boundary=e.snapshot_state();const auto layers=e.action_layers_;
    std::vector<std::unordered_map<uint64_t,double>> observations;
    std::vector<float> first_logits,logits;
    for(size_t mode=0;mode<=layers.size();++mode) {
        e.restore_state(boundary);e.kvmem_truncate_to(boundary.position);
        e.action_layers_=mode<layers.size()?std::vector<int32_t>{layers[mode]}:layers;
        e.kvmem_action_begin(100,101,102,103,false);
        ck(e.forward_n_tokens({200,201,202,203,204,205,206,207},true).ok,"equivalence forward");e.kvmem_register_append(8);
        ck(e.action_round_samples_==8,"equivalence sampled rows");
        ck(e.copy_last_logits(logits),"equivalence logits");
        if(!mode)first_logits=logits;else ck(first_logits==logits,"layer capture changed full-model logits");
        observations.push_back(e.action_round_mass_);e.kvmem_action_end();
    }
    for(const auto &[key,mass]:observations.back()) {
        double expected=0;for(size_t i=0;i<layers.size();++i)expected+=observations[i][key]/layers.size();
        ck(std::abs(expected-mass)<2e-6,"combined layer mean differs from individual layer captures");
    }
    std::cout<<"PASS real-model layer mean equals independent single-layer captures; logits bit exact\n";
}
static void run(const QwenNativeModel &model,const QwenWeights &weights,DeviceBackend &backend,const KvMemAttentionConfig &attention) {
    auto ck=[](bool ok,const char *s){if(!ok) throw std::runtime_error(s);};
    QwenExecutor e(model,weights,backend,4096);
    KvMemStoreConfig cfg;cfg.block_tokens=32;cfg.select_budget=cfg.prefill_budget=192;
    cfg.gen_budget=1024;cfg.sink_blocks=1;cfg.recent_blocks=0;
    cfg.immutable_source_k=true;cfg.cpu_tier_bytes=1ull<<30;cfg.gpu_memory_ratio=1;
    cfg.attention=attention;
    e.configure_kvmem(cfg);e.set_kvmem_enabled(true);e.kvmem_set_query_span(496,512,512);
    std::vector<uint32_t> history(512);for(uint32_t i=0;i<512;++i)history[i]=100+i%300;
    ck(e.forward_n_tokens(history,false).ok,"prefill");e.kvmem_register_append(history.size());
    auto boundary=e.snapshot_state();
    std::vector<uint32_t> ids={0,4,5,9,10,14};e.kvmem_set_selection(ids);
    e.kvmem_begin_query_replay(boundary,ids,false,true);
    ck(e.forward_n_tokens({100,101,102,103},true).ok,"query replay");e.kvmem_end_query_replay();e.kvmem_register_append(4);
    // Artificial marker IDs let this test exercise exact transition cases using
    // the real model/backend without depending on a sampled model response.
    auto saved=e.snapshot_state();
    // Match a real warm tool turn: restore the sparse M checkpoint, truncate
    // durable history/index caches, then append without another selection.
    const auto window_ids=e.bs_window_block_ids_;
    const auto window_bases=e.bs_win_base_host_;
    const auto window_lengths=e.bs_blk_tokens_host_;
    const auto warm_scores=e.kvmem_action_pressure_scores();
    e.restore_state(saved);e.kvmem_truncate_to(saved.position);
    ck(e.bs_window_block_ids_==window_ids && e.bs_win_base_host_==window_bases &&
       e.bs_blk_tokens_host_==window_lengths,"warm truncate lost sparse window identities");
    ck(e.kvmem_action_pressure_scores()==warm_scores,"warm truncate changed resident candidates");
    e.kvmem_action_begin(100,101,102,103,false);
    const std::vector<uint32_t> warm_action={100,200,201};
    e.kvmem_action_prepare(warm_action.data(),warm_action.size(),e.position());
    for(uint32_t w=0;w<window_ids.size();++w)
        ck(e.action_history_[w]==e.block_store_->blocks()[window_ids[w]].orig_pos_start,
           "warm action capture credited recent history instead of sparse key origin");
    e.kvmem_action_end();
    std::vector<DeviceArgmax> verified;
    std::vector<float> baseline_logits, captured_logits;
    ck(e.forward_n_tokens({100,200,201,202,101,102,200,103},true,&verified).ok,"baseline verifier");
    ck(e.copy_last_logits(baseline_logits),"baseline logits");e.restore_state(saved);
    e.kvmem_action_begin(100,101,102,103,false);
    ck(e.forward_n_tokens({100,200,201,202,101,102,200,103},true,&verified).ok,"verify");
    ck(e.copy_last_logits(captured_logits),"captured logits");
    ck(baseline_logits==captured_logits,"attention observation changed full-model logits");
    ck(e.action_round_samples_==0,"uncommitted verifier polluted history");
    // Reject suffix, replay accepted prefix through normal target decoding.
    e.restore_state(saved);
    ck(e.forward_n_tokens({100,200,201},true).ok,"accepted replay");
    e.kvmem_register_append(3);
    ck(e.action_round_samples_==1 && e.action_parser_.committed.tokens==2,"accepted count");
    ck(e.forward_one_token(202,true).ok,"plain action continuation");e.kvmem_register_append(1);
    ck(e.action_round_samples_==2,"plain continuation sample");
    ck(e.forward_n_tokens({101,102,100,200,201,103,200},true).ok,"excluded thinking");e.kvmem_register_append(7);
    ck(e.action_round_samples_==2,"thinking/tool marker exclusion");
    QwenExecutor::StateCheckpointSet checkpoints;
    const auto cp_base=e.position();
    ck(e.forward_n_tokens({100,200,201,202,203,204,205,101},true,nullptr,&checkpoints,8).ok,"checkpoint verifier");
    e.restore_state_checkpoint(checkpoints,2); // retain three target input rows
    ck(e.position()==cp_base+3,"checkpoint position");e.kvmem_register_append(3);
    ck(e.action_round_samples_==3,"rejected checkpoint suffix contributed samples");
    const auto pressure_before=e.block_store_->blocks()[0].action_score;
    auto live_scores=e.kvmem_action_pressure_scores();
    ck(!live_scores.empty(),"missing active action pressure scores");
    double live_mass=0;
    for(const auto &[id,score]:live_scores) {
        live_mass+=score;
        ck(id==0 || id==4 || id==5 || id==9 || id==10 || id==14 || id>=16,
           "pressure candidate includes cold historical block");
    }
    ck(std::abs(live_mass-1)<1e-5,"active mean including new tail");
    ck(e.block_store_->blocks()[0].action_score==pressure_before,"pressure overlay mutated history");
    auto live_scores_again=e.kvmem_action_pressure_scores();
    ck(live_scores==live_scores_again,"pressure overlay is not idempotent");
    // Private generation uses the same guard as guided Query generation and
    // cannot contribute or alter the live overlay. This fixture manually
    // installs windows, so it has no Query-index readiness prerequisite.
    const auto samples_before_probe=e.action_round_samples_;
    auto probe_boundary=e.snapshot_state();
    ck(e.kvmem_begin_bounded_private_generation_probe(32)>=3,"private probe start");
    ck(e.forward_n_tokens({100,200,201},true).ok,"private probe forward");
    ck(e.action_round_samples_==samples_before_probe,"private query polluted action history");
    ck(e.kvmem_action_pressure_scores().empty(),"pressure candidates active inside private query");
    e.kvmem_cancel_guided_query_probe();e.restore_state(probe_boundary);
    ck(e.kvmem_action_pressure_scores()==live_scores,"private query changed action overlay");
    e.kvmem_action_end();double sum=0;
    for(const auto &b:e.block_store_->blocks())sum+=b.action_score;
    ck(std::abs(sum-1)<1e-5,"history aggregation including new tail");
    ck(e.block_store_->blocks()[16].action_score>0,"new appended context got no attention credit");
    const auto removed=e.block_store_->blocks()[5].action_score;
    const auto previous=e.block_store_->blocks()[0].action_score;
    // Reselect to a different slot ordering and accumulate another round.
    // Manual remap at a page boundary, matching production reselect alignment.
    while(e.position()%e.kv_page_size()) {
        ck(e.forward_one_token(200,false).ok,"align remap");e.kvmem_register_append(1);
    }
    live_scores=e.kvmem_action_pressure_scores();
    const auto pressure_selected=e.block_store_->pick_prefill_pressure_blocks(
        e.kvmem_materialize_mandatory_blocks(),live_scores);
    const auto committed_score=e.block_store_->blocks()[0].action_score;
    e.kvmem_reselect_prefill_pressure();
    ck(e.bs_window_block_ids_==pressure_selected,"real pressure install differs from action selection");
    ck(e.window_query_pos_<=192,"real pressure exceeded fixed budget");
    ck(e.block_store_->blocks()[0].action_score==committed_score,"pressure install double-committed history");
    e.kvmem_set_selection({0,2,4,7,10,14});
    e.kvmem_action_begin(100,101,102,103,false);
    ck(e.forward_n_tokens({100,200,201},true).ok,"second round");e.kvmem_register_append(3);
    e.kvmem_action_end();ck(e.block_store_->blocks()[0].action_score>=previous*.5,"decay mapping");
    ck(e.block_store_->blocks()[5].action_score==removed*.5,"removed block confused with packed slot");
    e.reset_state();ck(!e.action_running_,"reset");
    e.kvmem_set_query_span(112,128,128);
    ck(e.forward_n_tokens(std::vector<uint32_t>(128,200),false).ok,"dense prefill");
    e.kvmem_register_append(128);ck(!e.kvmem_active_,"expected dense window");
    e.kvmem_action_begin(100,101,102,103,false);
    ck(e.forward_n_tokens({100,200,201,202},true,&verified).ok,"dense verifier");
    e.kvmem_register_append(4);e.kvmem_action_end();
    sum=0;for(const auto &b:e.block_store_->blocks())sum+=b.action_score;
    ck(std::abs(sum-1)<1e-5,"dense history aggregation");
    ck(!e.kvmem_action_pressure_scores().empty(),"completed round pressure candidates");
    std::cout<<"PASS real-model action capture: verifier discard/replay, accepted counts, plain continuation, thinking exclusion, round commit, reselection, reset\n";
}
static void run_non_thinking(const QwenNativeModel &model,const QwenWeights &weights,DeviceBackend &backend,const KvMemAttentionConfig &attention) {
    auto ck=[](bool ok,const char *s){if(!ok)throw std::runtime_error(s);};
    QwenExecutor e(model,weights,backend,4096);
    KvMemStoreConfig cfg;cfg.block_tokens=32;cfg.select_budget=cfg.prefill_budget=192;
    cfg.gen_budget=1024;cfg.sink_blocks=1;cfg.recent_blocks=0;
    cfg.immutable_source_k=true;cfg.cpu_tier_bytes=1ull<<30;cfg.gpu_memory_ratio=1;
    cfg.attention=attention;
    e.configure_kvmem(cfg);e.set_kvmem_enabled(true);
    ck(e.action_parser_.scope==detail::ActionAttentionParser::Scope::NonThinking,"scope configuration");
    ck(e.forward_n_tokens(std::vector<uint32_t>(128,200),false).ok,"non-thinking prompt prefill");
    e.kvmem_register_append(128);
    ck(e.action_round_samples_==0 && !e.action_running_,"prompt prefill was sampled");
    auto saved=e.snapshot_state();
    const std::vector<uint32_t> draft={200,103,210,211,100,212,101,213,102,220,103,214};
    std::vector<float> baseline,observed;
    QwenExecutor::StateCheckpointSet checkpoints;
    ck(e.forward_n_tokens(draft,true,nullptr,&checkpoints,draft.size()).ok &&
       e.copy_last_logits(baseline),"non-thinking baseline");
    e.restore_state(saved);
    e.kvmem_action_begin(100,101,102,103,true);
    ck(e.forward_n_tokens(draft,true,nullptr,&checkpoints,draft.size()).ok &&
       e.copy_last_logits(observed),"non-thinking verifier");
    ck(baseline==observed,"non-thinking capture changed logits");
    ck(e.action_round_samples_==0,"speculative output committed early");
    // Accept just a reasoning token: rejected think-close must not enable capture.
    e.restore_state_checkpoint(checkpoints,0);e.kvmem_register_append(1);
    ck(e.action_round_samples_==0 && e.action_parser_.committed.thinking,"rejected think-close state");
    ck(e.forward_n_tokens({200,201},true).ok,"accepted reasoning");e.kvmem_register_append(2);
    ck(e.action_round_samples_==0,"reasoning was sampled");
    ck(e.forward_n_tokens({103,210,211},true).ok,"ordinary answer");e.kvmem_register_append(3);
    ck(e.action_round_samples_==1 && e.action_parser_.committed.tokens==2,"ordinary output not sampled");
    ck(e.forward_n_tokens({212,102,220,103,213},true,nullptr,&checkpoints,5).ok,"rejected think-open verifier");
    e.restore_state_checkpoint(checkpoints,0);e.kvmem_register_append(1);
    ck(!e.action_parser_.committed.thinking && e.action_round_samples_==2,"rejected think-open state");
    ck(e.forward_n_tokens({214,215},true).ok,"answer continuation");e.kvmem_register_append(2);
    ck(e.action_round_samples_==3 && e.action_parser_.committed.tokens==5,"accepted answer count");
    ck(e.forward_n_tokens({100,216,217,101,218},true).ok,"tool and prose");e.kvmem_register_append(5);
    ck(e.action_round_samples_==4 && e.action_parser_.committed.tokens==8,"shared output counter");
    const auto before_probe=e.kvmem_action_pressure_scores();
    const auto probe_boundary=e.snapshot_state();
    ck(e.kvmem_begin_bounded_private_generation_probe(32)>=3,"non-thinking private probe start");
    ck(e.forward_n_tokens({219,220,221},true).ok,"non-thinking private query");
    ck(e.action_round_samples_==4 && e.action_parser_.committed.tokens==8,"private prose polluted statistics");
    e.kvmem_cancel_guided_query_probe();e.restore_state(probe_boundary);
    ck(e.kvmem_action_pressure_scores()==before_probe,"private query changed output pressure scores");
    e.kvmem_action_end();double sum=0;
    for(const auto &b:e.block_store_->blocks())sum+=b.action_score;
    ck(std::abs(sum-1)<1e-5,"non-thinking normalized round mass");
    const auto history=e.kvmem_action_pressure_scores();
    e.kvmem_action_begin(-1,-1,102,103,true);
    ck(e.forward_n_tokens({220,221},true).ok,"thinking-only response");e.kvmem_register_append(2);
    e.kvmem_action_end();
    ck(e.kvmem_action_pressure_scores()==history,"thinking-only response decayed history");
    e.kvmem_action_begin(-1,-1,102,103,false);
    ck(e.forward_one_token(222,true).ok,"no-tool answer");e.kvmem_register_append(1);
    ck(e.action_round_samples_==1,"no-tool output missing");e.kvmem_action_end();
    sum=0;for(const auto &b:e.block_store_->blocks())sum+=b.action_score;
    ck(std::abs(sum-1.5)<1e-5,"non-thinking history decay");
    std::cout<<"PASS real-model non-thinking capture: prompt/private exclusion, ordinary/tool output, rejected think boundaries, logits parity, pressure overlay, decay\n";
}
static void run_prefill(const QwenNativeModel &model,const QwenWeights &weights,DeviceBackend &backend,const KvMemAttentionConfig &attention) {
    auto ck=[](bool ok,const char *s){if(!ok)throw std::runtime_error(s);};
    const uint64_t sample_factor=attention.prefill_stride?1+(128-1)/attention.prefill_stride:1;
    QwenExecutor e(model,weights,backend,32768);
    KvMemStoreConfig cfg;cfg.block_tokens=32;cfg.select_budget=cfg.prefill_budget=192;
    cfg.gen_budget=1024;cfg.sink_blocks=1;cfg.recent_blocks=0;
    cfg.immutable_source_k=true;cfg.cpu_tier_bytes=1ull<<30;cfg.gpu_memory_ratio=1;
    cfg.attention=attention;
    e.configure_kvmem(cfg);e.set_kvmem_enabled(true);e.set_prefill_chunk_override(128);
    ck(e.prefill_attention_enabled_,"prefill configuration");
    std::vector<uint32_t> prompt(512);for(unsigned i=0;i<prompt.size();++i)prompt[i]=200+i%50;
    std::vector<float> baseline,observed;
    e.prefill_attention_enabled_=false;
    ck(e.forward_n_tokens(prompt,true).ok && e.copy_last_logits(baseline),"prefill baseline");
    e.reset_state();e.prefill_attention_enabled_=true;
    ck(e.forward_n_tokens(prompt,true).ok && e.copy_last_logits(observed),"prefill capture");
    e.kvmem_register_append(prompt.size());
    ck(baseline==observed,"prefill capture changed model logits");
    ck(e.prefill_attention_samples_==4*sample_factor && e.prefill_attention_end_==512,"last-row prefill samples");
    ck(std::abs(e.prefill_attention_scores_.at(480).exposure-1.0/sample_factor)<1e-12,
       "new last block credited queries that cannot causally see it");
    const double d=std::exp2(-128.0/attention.prefill_half_life);
    ck(std::abs(e.prefill_attention_scores_.at(0).exposure-(1+d+d*d+d*d*d))<1e-10,
       "prefill exposure/decay depends on query or layer sampling density");
    ck(e.action_round_samples_==0 && !e.action_running_,"input polluted output statistics");
    auto scores=e.kvmem_action_pressure_scores();
    ck(scores.size()==16,"prefill resident score count");
    for(const auto &[id,score]:scores) {
        ck(score==e.prefill_attention_scores_.at(id*32).mean(),"pressure did not use prefill mean");
        ck(e.block_store_->blocks()[id].action_score==0,"prefill changed action history");
    }
    ck(std::any_of(scores.begin(),scores.end(),[](auto x){return x.second>0;}),"prefill scores all zero");
    // At the first semantic boundary, no generated output exists. Use the
    // actual captured input means for the attention half of the fixed budget.
    std::vector<double> query(16);for(unsigned i=0;i<16;++i)query[i]=i;
    e.block_store_->set_retrieval_scores(query);
    std::vector<uint32_t> ranked;
    for(unsigned i=1;i<13;++i)ranked.push_back(i); // sink 0, Query 13/14/15 already kept
    std::sort(ranked.begin(),ranked.end(),[&](auto a,auto b){
        const auto sa=e.prefill_attention_scores_.at(a*32).mean(), sb=e.prefill_attention_scores_.at(b*32).mean();
        return sa!=sb?sa>sb:a>b;
    });
    std::vector<uint32_t> mixed={0,13,14,15,ranked[0],ranked[1]};std::sort(mixed.begin(),mixed.end());
    e.action_pressure_=false; // semantic fallback does not require pressure mode
    ck(e.kvmem_selection_with_pin()==mixed,"cold semantic attention quota not used");
    e.prefill_attention_enabled_=false;
    ck(e.kvmem_selection_with_pin()==std::vector<uint32_t>({0,11,12,13,14,15}),"disabled input changed Query fallback");
    e.prefill_attention_enabled_=true;e.action_pressure_=true;
    std::vector<double> output(16,0);output[12]=1;output[11]=2;
    e.block_store_->update_action_scores(output,0);
    ck(e.kvmem_selection_with_pin()==std::vector<uint32_t>({0,11,12,13,14,15}),"input overrode output semantic history");
    output.assign(16,0);e.block_store_->update_action_scores(output,0);
    const auto expected=e.block_store_->pick_prefill_pressure_blocks({},scores);
    e.kvmem_reselect_prefill_pressure();
    ck(e.bs_window_block_ids_==expected && e.window_query_pos_==192,"prefill pressure selection not installed");
    auto boundary=e.snapshot_state();
    ck(e.forward_n_tokens(std::vector<uint32_t>(128,220),false).ok,"sparse prefill append");
    e.kvmem_register_append(128);
    ck(e.prefill_attention_samples_==5*sample_factor,"sparse prefill sample missing");
    for(const auto &[key,score]:e.prefill_attention_scores_)
        ck(key>=512 || std::find(expected.begin(),expected.end(),key/32)!=expected.end(),"prefill score refers to evicted history");
    e.kvmem_action_commit(e.position());
    ck(e.prefill_attention_samples_==5*sample_factor,"prefill commit not idempotent");
    const auto saved=e.snapshot_state();scores=e.kvmem_action_pressure_scores();
    ck(e.kvmem_begin_bounded_private_generation_probe(32)>=3,"prefill private probe start");
    ck(e.forward_n_tokens({220,221,222},true).ok,"prefill private probe");
    e.kvmem_cancel_guided_query_probe();e.restore_state(saved);
    ck(e.prefill_attention_samples_==5*sample_factor && e.kvmem_action_pressure_scores()==scores,"private query changed prefill evidence");
    // Replay/recompute/provisional rows must never update input evidence.
    uint32_t token=222;
    for(bool *guard:{&e.kvmem_query_replay_active_,&e.kvmem_blend_active_,&e.kvmem_provisional_prefill_}) {
        *guard=true;e.kvmem_action_prepare(&token,1,e.position(),true);
        e.kvmem_action_commit(e.position()+1);*guard=false;
        ck(e.prefill_attention_samples_==5*sample_factor,"guarded replay counted as new input");
    }
    e.restore_state(boundary);e.kvmem_truncate_to(boundary.position);
    ck(e.prefill_attention_scores_.empty(),"rewritten history retained future prefill evidence");
    ck(e.forward_n_tokens(std::vector<uint32_t>(128,230),false).ok,"prefill after rewind");e.kvmem_register_append(128);
    const auto samples=e.prefill_attention_samples_;
    e.kvmem_action_begin(100,101,102,103,false);
    ck(e.forward_n_tokens({230,231,232},true).ok,"output after prefill");e.kvmem_register_append(3);
    e.kvmem_action_end();
    ck(e.prefill_attention_samples_==samples,"output was also counted as prefill");
    for(const auto &[id,score]:e.kvmem_action_pressure_scores())
        ck(score==e.block_store_->blocks()[id].action_score,"prefill mixed into nonzero action scores");
    e.reset_state();ck(e.prefill_attention_scores_.empty() && !e.prefill_attention_samples_,"prefill reset");
    // One long call spans actual capacity recoveries, not just explicit remaps.
    const auto generation=e.kvmem_selection_generation_;
    ck(e.forward_n_tokens(std::vector<uint32_t>(16384,200),false).ok,"long pressure prefill");
    e.kvmem_register_append(16384);
    ck(e.kvmem_selection_generation_>generation,"long prefill never recovered capacity");
    ck(e.prefill_attention_samples_==128*sample_factor && e.prefill_attention_end_==16384,"long prefill acceptance lost samples");
    ck(e.prefill_attention_scores_.size()<=e.kvmem_gpu_page_pool_->total_pages()*e.kv_page_size()/32+1,"unbounded prefill score state");
    e.reset_state();e.prefill_attention_interval_=1u<<20;
    ck(e.forward_n_tokens(std::vector<uint32_t>(4096,200),false).ok,"forced near-capacity sampling");
    e.kvmem_register_append(4096);
    ck(e.prefill_attention_samples_>sample_factor && e.prefill_attention_samples_<32*sample_factor,"capacity sampling did not override sparse interval");
    std::cout<<"PASS real-model prefill capture: last row beyond 64, logits parity, cold semantic quota/output priority, cold pressure, sparse identities, private/replay exclusion, rewind/reset, output separation, repeated capacity recovery\n";
}
};
}
int main(int argc,char**argv)try {
    if(argc<2)throw std::invalid_argument("usage: model-probe MODEL [--bench|--bench-prefill|--sweep-prefill|--sweep-output] [--layers POLICY] [--stride N]");
    qw3::KvMemAttentionConfig attention;attention.enabled=true;attention.pressure=true;attention.interval=2;
    std::string mode;
    for(int i=2;i<argc;++i) {
        const std::string arg=argv[i];
        if(arg=="--layers" && i+1<argc)attention.layers=argv[++i];
        else if(arg=="--stride" && i+1<argc)attention.prefill_stride=std::stoul(argv[++i]);
        else if(arg=="--bench" || arg=="--bench-prefill" || arg=="--sweep-prefill" || arg=="--sweep-output" || arg=="--equivalence" || arg=="--sweep-prefill-dense")mode=arg;
        else throw std::invalid_argument("unknown model probe argument");
    }
    attention.validate();
    setenv("QW3_PAGED_KV_PAGE_SIZE","16",1);
    auto backend=qw3::make_cuda_device_backend(qw3::LinearBackend::Auto);
    if(!backend || !backend->begin().ok)throw std::runtime_error("no CUDA backend");
    qw3::QwenNativeModel model(qw3::open_model_source(argv[1]));qw3::QwenWeights weights(model,*backend,false,false);
    if(mode=="--equivalence") {
        qw3::KvPageTableTestAccess::layer_equivalence(model,weights,*backend,attention);
    } else if(mode=="--sweep-prefill" || mode=="--sweep-output" || mode=="--sweep-prefill-dense") {
        qw3::KvPageTableTestAccess::bench_sampling(model,weights,*backend,attention,mode!="--sweep-output",mode=="--sweep-prefill-dense");
    } else if(mode=="--bench-prefill") {
        attention.prefill=true;attention.prefill_interval=8192;
        qw3::KvPageTableTestAccess::bench_prefill(model,weights,*backend,attention);
    } else if(mode=="--bench") qw3::KvPageTableTestAccess::bench(model,weights,*backend,attention);
    else {
        attention.prefill=false;attention.scope="tool-call";
        qw3::KvPageTableTestAccess::run(model,weights,*backend,attention);
        attention.scope="non-thinking";
        qw3::KvPageTableTestAccess::run_non_thinking(model,weights,*backend,attention);
        attention.prefill=true;attention.prefill_interval=128;attention.fraction=.5;
        qw3::KvPageTableTestAccess::run_prefill(model,weights,*backend,attention);
    }
    if(!backend->end().ok)throw std::runtime_error("backend end");
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
