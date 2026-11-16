#include "qwen_executor.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>

namespace qw3 {
struct KvPageTableTestAccess {
static void run(const QwenNativeModel &model,const QwenWeights &weights,DeviceBackend &backend,
                const std::string &log,uint32_t tokens,uint32_t budget,unsigned repeats) {
    std::vector<float> reference;
    std::vector<uint32_t> selection;
    for (unsigned trial=0;trial<repeats;++trial) for (unsigned order=0;order<2;++order) {
        const bool observe=((trial+order)%2)!=0;
        QwenExecutor e(model,weights,backend,std::max(65536u,tokens+2048));
        KvMemStoreConfig c;c.block_tokens=32;c.select_budget=c.prefill_budget=budget;
        c.gen_budget=16384;c.sink_blocks=std::min(41u,budget/64);c.recent_blocks=0;
        c.immutable_source_k=true;c.index_placement=KvMemIndexPlacement::CPU;
        c.cpu_tier_bytes=8ull<<30;c.gpu_memory_ratio=.7;
        if (observe) { c.mismatch.log_path=log; c.mismatch.interval=256;c.mismatch.query_mode="block-mean"; }
        e.configure_kvmem(c);e.set_kvmem_enabled(true);e.set_prefill_chunk_override(1024);
        e.kvmem_set_query_span(tokens-32,tokens,tokens);
        std::vector<uint32_t> history(tokens);
        for (uint32_t i=0;i<tokens;++i) history[i]=200+(i/64)%200;
        auto check=[](bool v,const char *s){if (!v) throw std::runtime_error(s);};
        backend.synchronize();auto start=std::chrono::steady_clock::now();
        check(e.forward_n_tokens(history,false).ok,"prefill failed");e.kvmem_register_append(tokens);
        backend.synchronize();double prefill=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        e.kvmem_reselect();
        if (reference.empty()) selection=e.bs_window_block_ids_;
        else check(e.bs_window_block_ids_==selection,"observer changed selected KV");
        e.kvmem_action_begin(100,101,102,103,false);
        // Force non-thinking output, then a thinking span, then ordinary output.
        // Interval is reduced only for this correctness probe's short response.
        if (observe) e.mismatch_parser_.interval=16;
        start=std::chrono::steady_clock::now();
        for (uint32_t i=0;i<128;++i) {
            const uint32_t token=i==32?102:i==64?103:210+i%16;
            check(e.forward_one_token(token,true).ok,"decode failed");
            e.kvmem_register_append(1);
        }
        e.kvmem_action_end();backend.synchronize();
        double decode=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::vector<float> logits(e.logits_->count);
        check(backend.copy_to_host(*e.logits_,logits.data(),0,logits.size()).ok,"logits read failed");
        if (reference.empty()) reference=logits;
        else { check(logits.size()==reference.size(),"logit shape changed");
            for (size_t i=0;i<logits.size();++i) check(logits[i]==reference[i],"observer changed output logits"); }
        if (observe) {
            check(e.mismatch_parser_.committed.tokens==128,"observer excluded thinking output");
            const auto progress=e.mismatch_parser_.committed.tokens;
            uint32_t token=210;
            for (bool *guard:{&e.kvmem_query_replay_active_,&e.kvmem_provisional_prefill_,&e.kvmem_blend_active_}) {
                *guard=true;e.kvmem_mismatch_prepare(&token,1,e.position(),false);
                e.kvmem_mismatch_commit(e.position()+1);*guard=false;
                check(e.mismatch_parser_.committed.tokens==progress,"private/replay guard failed");
            }
            // Probe a speculative four-row transaction; only its accepted
            // prefix may be submitted, and a repeated commit must be inert.
            e.kvmem_action_begin(100,101,102,103,false);
            e.mismatch_parser_.interval=1;
            const auto boundary=e.snapshot_state();
            const auto before=e.mismatch_parser_.committed.tokens;
            std::vector<DeviceArgmax> verified;
            check(e.forward_n_tokens({210,211,212,213},true,&verified).ok,"verifier batch failed");
            e.kvmem_mismatch_commit(boundary.position+2);
            check(e.mismatch_parser_.committed.tokens==before+2,"unaccepted rows counted");
            e.kvmem_mismatch_commit(boundary.position+2);
            check(e.mismatch_parser_.committed.tokens==before+2,"duplicate commit counted");
            e.restore_state(boundary);
            e.kvmem_action_end();
            e.mismatch_observer_->finish();
        }
        std::cout<<"MISMATCH_BENCH observe="<<observe<<" trial="<<trial<<" tokens="<<tokens
            <<" budget="<<budget<<" prefill_s="<<prefill<<" prefill_tps="<<tokens/prefill
            <<" decode_s="<<decode<<" decode_tps="<<128/decode<<" logits_identical=1 selection_identical=1\n"<<std::flush;
    }
}
};
}
int main(int argc,char **argv) try {
    if (argc<3) throw std::runtime_error("usage: mismatch-model-probe MODEL LOG [TOKENS BUDGET REPEATS]");
    auto backend=qw3::make_cuda_device_backend(qw3::LinearBackend::Auto);
    if (!backend || !backend->begin().ok) throw std::runtime_error("no CUDA backend");
    qw3::QwenNativeModel model(qw3::open_model_source(argv[1]));
    qw3::QwenWeights weights(model,*backend,false,false);
    qw3::KvPageTableTestAccess::run(model,weights,*backend,argv[2],
        argc>3?std::stoul(argv[3]):8192,argc>4?std::stoul(argv[4]):4096,argc>5?std::stoul(argv[5]):2);
    if (!backend->end().ok) throw std::runtime_error("backend end failed");
    return 0;
} catch (const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
