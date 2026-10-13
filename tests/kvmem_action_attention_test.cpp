#include "kvmem_action_attention.hpp"
#include "qw3/kvmem_store.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace qw3;
static void check(bool x, const char *why="action attention test failed") { if (!x) throw std::runtime_error(why); }
static void input_history_identity() {
    KvMemStoreConfig c;c.block_tokens=32;c.select_budget=64;
    c.sink_blocks=c.recent_blocks=0;c.action_fraction=.5;
    auto repo=std::make_shared<KvMemRepository>(KvMemRepository::Config{3,32});
    KvMemStore s(c,repo);
    const uint64_t base=uint64_t{1}<<34;
    std::vector<KvMemBlockHandle> handles;
    for(unsigned i=0;i<3;++i) {
        auto lease=repo->try_reserve(base+i*32,32).commit();
        handles.push_back(lease.handle());
        s.register_repository_block(std::move(lease),i*32);
        s.set_block_tier(handles.back().slot,KvTier::CPU,i,-1,handles.back().generation);
    }
    s.set_repository_retrieval_scores(handles,{0,1,2});
    const KvMemStore::PrefillSelectionScores input={{base,10}};
    check(s.pick_topk_blocks({},input)==std::vector<uint32_t>({0,2}),"64-bit input score identity");
    check(repo->retire_if_unpinned(handles[0]),"retire input-scored block");
    check(repo->try_reclaim(handles[0],[](const auto&){return true;})==KvMemRepository::ReclaimResult::Reclaimed,"reclaim input-scored block");
    auto next=repo->try_reserve(base+96,32).commit();
    handles[0]=next.handle();s.register_repository_block(std::move(next),96);
    s.set_repository_retrieval_scores(handles,{0,1,2});
    check(s.pick_topk_blocks({},input)==std::vector<uint32_t>({1,2}),"reused slot inherited old input score"); // no credit from old slot 0
    check(s.pick_topk_blocks({},{{base+96,10}})==std::vector<uint32_t>({2,0}),"history order differs from slot order"); // history order, not slot order
}
static void non_thinking_parser() {
    using Parser=detail::ActionAttentionParser;
    check(Parser::parse_scope("tool-call")==Parser::Scope::ToolCall);
    check(Parser::parse_scope("non-thinking")==Parser::Scope::NonThinking);
    bool rejected=false;
    try { Parser::parse_scope("thinking"); } catch(const std::invalid_argument&) { rejected=true; }
    check(rejected);
    Parser p;
    p.scope=Parser::Scope::NonThinking;
    p.open=1;p.close=2;p.think_open=3;p.think_close=4;p.interval=2;
    // Prose, code and tool arguments share a counter; all markers and reasoning
    // are excluded. The initial think-open may already be in the prompt.
    p.committed.thinking=true;
    std::vector<uint32_t> t={10,4,11,12,1,13,2,14,3,15,4,16};
    p.prepare(t.data(),t.size());
    check(p.mask==((1ull<<2)|(1ull<<5)|(1ull<<11)));
    check(p.commit(t.size())==p.mask && p.committed.tokens==5);
    check(p.commit(t.size())==0);
    // A rejected think-open must not hide subsequent accepted output.
    t={17,3,18};p.prepare(t.data(),t.size());check(p.commit(1)==0);
    check(!p.committed.thinking && p.committed.tokens==6);
    t={19};p.prepare(t.data(),1);check(p.commit(1)==1);
    // A rejected think-close must not count speculative prose as actual output.
    t={3,20,4,21,22};p.prepare(t.data(),t.size());check(p.commit(2)==0);
    check(p.committed.thinking && p.committed.tokens==7);
    t={23,24};p.prepare(t.data(),t.size());check(p.commit(2)==0);
    t={4,25,26};p.prepare(t.data(),t.size());check(p.commit(3)==(1ull<<2));
    check(p.committed.tokens==9 && !p.committed.thinking);
    // Sequential and batched consumption must agree for every accepted prefix.
    const std::vector<uint32_t> stream={10,1,11,12,2,13,3,14,1,15,2,4,16,17,18};
    for(size_t accepted=0;accepted<=stream.size();++accepted) {
        Parser batch=p, serial=p;
        batch.prepare(stream.data(),stream.size());
        auto mask=batch.commit(accepted);
        uint64_t serial_mask=0;
        for(size_t i=0;i<accepted;++i) {
            serial.prepare(&stream[i],1);
            if(serial.commit(1)) serial_mask|=1ull<<i;
        }
        check(mask==serial_mask && batch.committed.tokens==serial.committed.tokens &&
              batch.committed.thinking==serial.committed.thinking &&
              batch.committed.action==serial.committed.action);
    }
    // Non-tool models can still collect ordinary output without tool delimiters.
    p.open=p.close=-1;
    t={27,28};p.prepare(t.data(),t.size());check(p.commit(2)==2);
}
static void fused_scores() {
    KvMemStoreConfig c;c.block_tokens=32;c.select_budget=64;
    c.sink_blocks=c.recent_blocks=0;c.action_fraction=.5;c.action_combine="score";
    auto pick=[&](const std::vector<double>&q,const std::vector<double>&a,double weight=.5) {
        auto config=c;config.action_fraction=weight;
        KvMemStore s(config);s.register_append(q.size()*32);
        s.set_retrieval_scores(q);s.update_action_scores(a,0);
        return s.pick_topk_blocks();
    };
    // Quotas choose block 0 by Query and block 2 by attention. Fused scores
    // (.2,.15,.3778,.2722) instead choose 2 and 3 in one global ranking.
    const std::vector<double> q={4,3,2,1}, a={0,0,5,4};
    check(pick(q,a)==std::vector<uint32_t>({2,3}),"fusion still behaves like quotas");
    auto quota=c;quota.action_combine="quota";
    KvMemStore old(quota);old.register_append(128);old.set_retrieval_scores(q);old.update_action_scores(a,0);
    check(old.pick_topk_blocks()==std::vector<uint32_t>({0,2}),"quota regression");
    check(pick({4e300,3e300,2e300,1e300},{0,0,5e-300,4e-300})==std::vector<uint32_t>({2,3}),"fusion is not scale invariant");
    check(pick(q,{0,0,0,0})==std::vector<uint32_t>({0,1}),"missing attention must fall back to Query");
    check(pick({0,0,0,0},a)==std::vector<uint32_t>({2,3}),"missing Query must fall back to attention");
    check(pick({0,0,0,0},{0,0,0,0})==std::vector<uint32_t>({2,3}),"zero-score tie must prefer recent blocks");
    check(pick({-1,-2,-3,-4},{0,0,0,0})==std::vector<uint32_t>({0,1}),"negative Query fallback changed ranking");
    check(pick(q,a,0)==std::vector<uint32_t>({0,1}),"zero attention weight must be Query only");
    check(pick(q,a,1)==std::vector<uint32_t>({2,3}),"full attention weight must be attention only");
    // Huge mandatory scores are excluded from normalization, not merely top-k.
    auto pinned=c;pinned.select_budget=128;pinned.sink_blocks=1;
    KvMemStore s(pinned);s.register_append(192);
    s.set_retrieval_scores({1e300,1e300,4,3,2,1});
    s.update_action_scores({1e300,1e300,0,0,5,4},0);
    check(s.pick_topk_blocks({1})==std::vector<uint32_t>({0,1,4,5}),"mandatory blocks distorted normalization");
    // Cold input scores remain separate; accepted output history wins thereafter.
    KvMemStore cold(c);cold.register_append(128);cold.set_retrieval_scores(q);
    check(cold.pick_topk_blocks({},{{64,5},{96,4},{99999,1e300}})==std::vector<uint32_t>({2,3}),"prefill fusion identity");
    cold.update_action_scores({0,1,0,0},0);
    check(cold.pick_topk_blocks({},{{64,5},{96,4}})==std::vector<uint32_t>({0,1}),"prefill displaced output history");
    bool rejected=false;
    try { auto bad=c;bad.action_combine="invalid";KvMemStore invalid(bad); }
    catch(const std::invalid_argument&) {rejected=true;}
    check(rejected,"invalid fusion mode accepted");
}
int main() {
    std::vector<int32_t> standard;
    for (int32_t i=3;i<64;i+=4) standard.push_back(i);
    check(detail::resolve_attention_layers("last",standard)==std::vector<int32_t>{63});
    check(detail::resolve_attention_layers("spread3",standard)==std::vector<int32_t>({3,31,63}));
    check(detail::resolve_attention_layers("all",standard)==standard);
    check(detail::resolve_attention_layers("63,3,31",standard)==std::vector<int32_t>({3,31,63}));
    check(detail::resolve_attention_layers("spread3",{3})==std::vector<int32_t>{3});
    check(detail::resolve_attention_layers("spread3",{3,7})==std::vector<int32_t>({3,7}));
    for (const auto &policy : {"", "3,,63", "-1", "3,3", "3,", "4", "64", "2147483648"}) {
        bool rejected=false;
        try {(void)detail::resolve_attention_layers(policy,standard);} catch (const std::invalid_argument&) {rejected=true;}
        check(rejected,"invalid attention layer policy accepted");
    }
    input_history_identity();
    non_thinking_parser();
    fused_scores();
    detail::PrefillAttentionScore old, fresh;
    for(unsigned i=0;i<100;++i) old.observe(.2,.5);
    fresh.observe(.2,.5);
    check(std::abs(old.mean()-fresh.mean())<1e-12); // exposure cannot inflate importance
    const double prior=old.mean();old.observe(0,.5);
    check(old.mean()<prior && old.mean()>0);
    detail::ActionAttentionParser p;
    p.open=1;p.close=2;p.think_open=3;p.think_close=4;p.interval=2;
    std::vector<uint32_t> t={3,1,5,4,1,6,7,8,2,9};
    p.prepare(t.data(),t.size()); check(p.mask==((1ull<<5)|(1ull<<7)));
    check(p.committed.tokens==0); // speculative preparation has no effect
    check(p.commit(7)==(1ull<<5)); check(p.committed.tokens==2);
    check(p.commit(7)==0); // idempotent
    t={10,2,11};p.prepare(t.data(),t.size()); check(p.mask==1);
    check(p.commit(3)==1);check(p.committed.tokens==3 && !p.committed.action);
    // Rejected closing markers must not close a subsequently accepted action.
    t={1,12,2};p.prepare(t.data(),t.size());p.commit(2);
    check(p.committed.action);t={13};p.prepare(t.data(),1);check(p.mask==1);p.commit(1);
    t={3,1,10,4,10};p.prepare(t.data(),t.size());check(p.mask==0);

    KvMemStoreConfig c;c.block_tokens=32;c.select_budget=5*32;c.sink_blocks=1;
    c.recent_blocks=0;c.action_fraction=.25;
    KvMemStore s(c);s.register_append(10*32);
    std::vector<double> q(10),h(10);for(size_t i=0;i<10;++i) q[i]=i;
    s.set_retrieval_scores(q);
    check(s.pick_topk_blocks()==std::vector<uint32_t>({0,6,7,8,9})); // no positive action -> Query
    const KvMemStore::PrefillSelectionScores input={{32,100},{64,50},{999999,1e9}};
    check(s.pick_topk_blocks({},input)==std::vector<uint32_t>({0,1,7,8,9}));
    check(s.blocks()[1].action_score==0); // input must not become output history
    check(s.pick_topk_blocks({},{{9*32,1}})==std::vector<uint32_t>({0,6,7,8,9})); // Query overlap fills from Query
    check(s.pick_topk_blocks({4},input)==std::vector<uint32_t>({0,4,7,8,9})); // rounded zero attention slots
    h[1]=1;s.update_action_scores(h,.5);
    check(s.pick_topk_blocks()==std::vector<uint32_t>({0,1,7,8,9}));
    h[1]=0;h[2]=1;s.update_action_scores(h,.5);
    check(s.blocks()[1].action_score==.5 && s.blocks()[2].action_score==1);
    check(s.pick_topk_blocks()==std::vector<uint32_t>({0,2,7,8,9}));
    check(s.pick_topk_blocks({},input)==std::vector<uint32_t>({0,2,7,8,9})); // output wins over larger input mass
    h.assign(10,0);h[9]=10;s.update_action_scores(h,0);
    check(s.pick_topk_blocks({},input)==std::vector<uint32_t>({0,6,7,8,9})); // Query overlap must not activate input fallback
    h.assign(10,0);h[1]=.5;h[2]=1;s.update_action_scores(h,0);
    KvMemStoreConfig balanced=c;balanced.action_fraction=.5;balanced.recent_blocks=1;
    KvMemStore cold(balanced);cold.register_append(10*32);cold.set_retrieval_scores(q);
    check(cold.pick_topk_blocks({},input)==std::vector<uint32_t>({0,1,7,8,9}));
    check(cold.pick_topk_blocks({5},input)==std::vector<uint32_t>({0,1,5,8,9}));
    check(cold.pick_topk_blocks({5})==std::vector<uint32_t>({0,5,7,8,9}));
    // Pressure ranking ignores Query scores, preserves mandatory fresh evidence,
    // and never fetches a cold block missing from the resident candidate list.
    check(s.pick_prefill_pressure_blocks()==std::vector<uint32_t>({0,6,7,8,9}));
    KvMemStore::PressureActionScores resident={{1,1},{2,2},{3,3},{4,4},{5,0},{6,0},{7,0},{8,0},{9,0}};
    check(s.pick_prefill_pressure_blocks({9},resident)==std::vector<uint32_t>({0,2,3,4,9}));
    resident={{1,0},{2,0},{3,0},{4,0},{5,0},{6,0}};
    check(s.pick_prefill_pressure_blocks({9},resident)==std::vector<uint32_t>({0,4,5,6,9}));
    resident={{1,1},{1,1},{2,2},{3,3},{4,4}};
    check(s.pick_prefill_pressure_blocks({},resident)==std::vector<uint32_t>({0,1,2,3,4}));
    check(s.blocks()[1].action_score==.5 && s.blocks()[2].action_score==1); // pure selector
    // Mandatory protection consumes the same fixed budget even with attention.
    bool rejected=false;
    try { s.pick_prefill_pressure_blocks({1,2,3,4,5},resident); }
    catch(const std::runtime_error&) { rejected=true; }
    check(rejected);
    std::cout << "PASS attention parsers; selection: fallback, quota, normalized score fusion, scale invariance, protected budget, source precedence, decay\n";
}
