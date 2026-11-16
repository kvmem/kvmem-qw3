#include "kvmem_mismatch_observer.hpp"
#include "kvmem_action_attention.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <sstream>

using namespace qw3::detail;
static void ck(bool value,const char *what) { if (!value) throw std::runtime_error(what); }
static bool near(double a,double b,double eps=1e-6) { return std::abs(a-b)<eps; }
static MismatchJob job() {
    MismatchJob j; j.layers={3}; j.heads=2; j.kv_heads=1; j.dim=1;
    j.block_tokens=32; j.position=j.progress=96; j.indexed_tokens=96;
    j.query_positions={95}; j.query_token_counts={32};j.queries={0x3c00,0x3c00}; j.tail_begin=96;
    j.selected_tokens={32,0,0}; j.live={1,1,1};
    for (uint16_t x:{0x0000,0x3c00,0x4000}) {
        auto b=std::make_shared<MismatchKeyBlock>(); b->keys={x}; j.blocks.push_back(b);
    }
    return j;
}
static MismatchScore score(const MismatchJob &j) {
    std::vector<std::shared_ptr<const MismatchKeyBlock>> ids;
    std::vector<std::vector<float>> values;
    return score_mismatch(j,ids,values);
}
static void normalized_tests() {
    auto j=job();
    // Uniform block mass is uniform token mass, including partial prefixes.
    j.mean_queries={0,0};j.selected_tokens={32,16,0};j.mandatory_tokens={16,0,0};
    auto s=score(j);
    ck(s.query_candidate_tokens==std::vector<uint64_t>{96} &&
       s.query_visible_tokens==std::vector<uint64_t>{48} &&
       s.query_mandatory_tokens==std::vector<uint64_t>{16},"uniform reference token counts");
    ck(near(s.query_mandatory_coverage[0],1.0/6) && near(s.query_random_coverage[0],.5) &&
       near(s.query_lift[0],1) && s.query_baseline_valid[0] &&
       s.query_baseline_status[0]=="ok","uniform token baseline gives unit lift");
    // A concentrated distribution may legitimately have lift greater than one.
    j=job();j.mean_queries={40,40};j.selected_tokens={0,0,32};
    s=score(j);
    ck(near(s.query_coverage[0],1) && near(s.random_coverage,1.0/3) && near(s.lift,3),
       "concentrated selected block beats uniform random slots");
    j.selected_tokens={32,0,0};
    ck(score(j).lift<1e-6,"concentrated omitted block has low lift");
    j=job();j.layers={3,63};j.mean_queries={1,-1,-1,-1};j.selected_tokens={32,16,0};
    j.mandatory_tokens={32,0,0};
    for (size_t b=0;b<3;++b) {
        auto key=std::make_shared<MismatchKeyBlock>();
        key->keys={j.blocks[b]->keys[0],j.blocks[b]->keys[0]};j.blocks[b]=key;
    }
    const double zp=1+std::exp(1.0)+std::exp(2.0), zn=1+std::exp(-1.0)+std::exp(-2.0);
    const double cp=(1+.5*std::exp(1.0))/zp, cn=(1+.5*std::exp(-1.0))/zn;
    const double ep=1/zp+.25*(1-1/zp), en=1/zn+.25*(1-1/zn);
    s=score(j);
    ck(near(s.query_mandatory_coverage[0],(.25/zp+.75/zn)) &&
       near(s.query_coverage[0],.25*cp+.75*cn) &&
       near(s.random_coverage,.25*ep+.75*en) &&
       near(s.lift,(.25*cp+.75*cn)/(.25*ep+.75*en)) &&
       std::abs(s.lift-(.25*cp/ep+.75*cn/en))>.01,
       "GQA heads and layers average C and mandatory mass before division");

    // Sink prefix and tail overlap within a partial block. Only their union
    // is fixed; the selected nonmandatory middle prefix remains randomizable.
    j=job();j.selected_tokens={32,16,16};j.mandatory_tokens={32,8,16};j.tail_begin=72;
    s=score(j);
    const double z=1+std::exp(1.0)+std::exp(2.0);
    const double cm=(1+.25*std::exp(1.0)+std::exp(2.0))/z;
    ck(s.query_candidate_tokens[0]==96 && s.query_visible_tokens[0]==80 &&
       s.query_mandatory_tokens[0]==72,"sink/partial/tail union token counts");
    ck(near(s.query_mandatory_coverage[0],cm) && near(s.random_coverage,cm+(1-cm)/3),
       "mandatory mass includes overlap exactly once");
    const auto with_mandatory=s;
    j.mandatory_tokens.clear();const auto without_mandatory=score(j);
    ck(with_mandatory.missing==without_mandatory.missing &&
       with_mandatory.query_coverage==without_mandatory.query_coverage &&
       with_mandatory.min_missing==without_mandatory.min_missing &&
       with_mandatory.max_missing==without_mandatory.max_missing,
       "normalization metadata must not alter original coverage/debt inputs");
    j.mandatory_tokens={32,8,16};j.live[1]=0;j.blocks[1].reset();
    s=score(j);
    ck(s.query_candidate_tokens[0]==64 && s.query_visible_tokens[0]==64 &&
       s.query_mandatory_tokens[0]==64 && s.query_baseline_status[0]=="no_selection" &&
       s.query_baseline_valid[0] && s.random_coverage==1 && s.lift==1,
       "unavailable history is excluded from all reference counts and mass");

    j=job();j.selected_tokens={32,32,32};s=score(j);
    ck(s.query_baseline_status[0]=="full_retention" && s.query_baseline_valid[0] &&
       s.random_coverage==1 && s.lift==1,"full retention baseline is one");
    j.mandatory_tokens={32,32,32};s=score(j);
    ck(s.query_baseline_status[0]=="no_selection" && s.query_baseline_valid[0] &&
       s.query_mandatory_coverage[0]==1 && s.random_coverage==1 && s.lift==1,
       "all mandatory has no random selection and unit reference");
    j=job();j.selected_tokens={0,0,0};s=score(j);
    ck(s.query_baseline_status[0]=="zero_expectation" && !s.query_baseline_valid[0] &&
       s.query_random_coverage[0]==0 && std::isnan(s.query_lift[0]) &&
       s.random_coverage==0 && std::isnan(s.lift),"zero expectation is not fabricated unit lift");
    j.live={0,0,0};s=score(j);
    ck(s.query_baseline_status[0]=="no_candidates" && !s.query_baseline_valid[0] &&
       s.query_candidate_tokens[0]==0 && s.query_visible_tokens[0]==0 &&
       s.query_mandatory_tokens[0]==0 && std::isnan(s.query_coverage[0]) &&
       std::isnan(s.query_mandatory_coverage[0]) && std::isnan(s.query_random_coverage[0]) &&
       std::isnan(s.random_coverage) && std::isnan(s.lift),"no candidates remains undefined");

    // Different queries have different causal candidate sets. A self-partial
    // block is excluded, and zero-candidate rows do not enter either mean.
    j=job();j.query_positions={15,31,63,95};j.queries.assign(8,0x3c00);
    j.selected_tokens={16,32,0};j.mandatory_tokens={8,0,0};s=score(j);
    ck(s.query_candidate_tokens==std::vector<uint64_t>({0,32,64,96}) &&
       s.query_visible_tokens==std::vector<uint64_t>({0,16,48,48}) &&
       s.query_mandatory_tokens==std::vector<uint64_t>({0,8,8,8}),"per-query causal token counts");
    double mean_c=0,mean_e=0,mean_l=0;
    for (size_t r=1;r<4;++r) {
        mean_c+=s.query_coverage[r]/3;mean_e+=s.query_random_coverage[r]/3;mean_l+=s.query_lift[r]/3;
    }
    ck(std::isnan(s.query_coverage[0]) && near(1-s.missing,mean_c) &&
       near(s.random_coverage,mean_e) && near(s.lift,mean_c/mean_e) &&
       std::abs(s.lift-mean_l)>.01,"aggregate lift is ratio of matched means, not mean lift");
    j=job();j.query_positions={47};s=score(j);
    ck(s.query_candidate_tokens[0]==32,"candidate tokens exclude partial future block");

    // Enumerate every size-k subset of optional individual token seats. Block
    // mass is apportioned uniformly; this oracle does not invoke score_mismatch.
    j=job();j.block_tokens=2;j.indexed_tokens=j.position=6;j.query_positions={5};
    j.selected_tokens={2,0,0};j.mandatory_tokens={1,0,0};j.tail_begin=5;s=score(j);
    const std::vector<double> token_mass={.5/z,.5/z,.5*std::exp(1.0)/z,.5*std::exp(1.0)/z,
                                         .5*std::exp(2.0)/z,.5*std::exp(2.0)/z};
    const std::vector<size_t> optional={1,2,3,4};
    const unsigned k=unsigned(s.query_visible_tokens[0]-s.query_mandatory_tokens[0]);
    double enumerated=0;unsigned combinations=0;
    for (unsigned mask=0;mask<(1u<<optional.size());++mask) {
        unsigned seats=0;for (size_t i=0;i<optional.size();++i) seats+=(mask>>i)&1;
        if (seats!=k) continue;
        double mass=token_mass[0]+token_mass[5];
        for (size_t i=0;i<optional.size();++i) if (mask&(1u<<i)) mass+=token_mass[optional[i]];
        enumerated+=mass;++combinations;
    }
    ck(combinations==4 && s.query_candidate_tokens[0]==6 && s.query_visible_tokens[0]==3 &&
       s.query_mandatory_tokens[0]==2 && near(s.random_coverage,enumerated/combinations),
       "reference expectation agrees with exact random token-subset enumeration");

    // Invalid mandatory prefixes must fail visibly instead of being clamped.
    for (unsigned invalid=0;invalid<3;++invalid) {
        j=job();j.mandatory_tokens=invalid==0?std::vector<uint32_t>{1}:
            invalid==1?std::vector<uint32_t>{33,0,0}:std::vector<uint32_t>{0,1,0};
        bool rejected=false;try {score(j);} catch (const std::runtime_error &) {rejected=true;}
        ck(rejected,"invalid mandatory prefix must be rejected");
    }
}
static void bench() {
    for (uint32_t tokens:{131072u,1048576u}) for (uint32_t samples:{1u,4u}) {
        MismatchJob j;j.layers={3,31,63};j.heads=24;j.kv_heads=4;j.dim=256;
        j.block_tokens=32;j.position=j.progress=j.indexed_tokens=tokens;j.tail_begin=tokens;
        j.query_positions.assign(samples,tokens-1);j.queries.assign(3*samples*24*256,0x3400);
        const uint32_t n=tokens/32;j.live.assign(n,1);j.selected_tokens.assign(n,0);
        for (uint32_t b=0;b<n;++b) {
            auto k=std::make_shared<MismatchKeyBlock>();k->keys.resize(3*4*256);
            for (size_t i=0;i<k->keys.size();++i) k->keys[i]=0x3000+(b+i)%1024;
            j.blocks.push_back(k);if (b<1152) j.selected_tokens[b]=32;
        }
        std::vector<std::shared_ptr<const MismatchKeyBlock>> ids;
        std::vector<std::vector<float>> values;
        for (uint32_t trial=0;trial<4;++trial) {
            const auto start=std::chrono::steady_clock::now();const auto result=score_mismatch(j,ids,values);
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            std::cout<<"CPU_PROBE_BENCH tokens="<<tokens<<" queries="<<samples<<" trial="<<trial
                <<" cpu_ms="<<ms<<" missing="<<result.missing<<"\n"<<std::flush;
        }
    }
}
int main(int argc,char **argv) try {
    if (argc>1 && std::string(argv[1])=="--bench") {bench();return 0;}
    normalized_tests();
    ck(near(mismatch_half_to_float(0x3c00),1) && near(mismatch_half_to_float(0xc000),-2),"half normal conversion");
    ck(mismatch_half_to_float(1)==std::ldexp(1.0f,-24),"half subnormal conversion");
    auto j=job(); const double z=1+std::exp(1.0)+std::exp(2.0);
    ck(near(score(j).missing,1-1/z),"global missing mass/GQA");
    j.mean_queries={1,1};j.queries.clear();
    ck(near(score(j).query_coverage.at(0),1/z),"FP32 averaged Query and per-block coverage");
    j.selected_tokens={32,32,32}; ck(near(score(j).missing,0),"full active history has no missing mass");
    j.selected_tokens={0,0,0}; ck(near(score(j).missing,1),"empty window misses everything");
    j.selected_tokens={32,16,0}; j.tail_begin=48;
    ck(near(score(j).missing,0),"partial selected block plus appended tail union");
    j.tail_begin=96; ck(near(score(j).missing,(0.5*std::exp(1.0)+std::exp(2.0))/z),"partial block coverage");
    j=job(); j.query_positions={47};
    ck(near(score(j).missing,0) && score(j).candidate_blocks==1,"no future/self-partial block leakage");
    j=job(); j.sink_blocks=1; ck(near(score(j).missing,1-1/z),"sink participates in global normalization");
    j.live[2]=0; j.selected_tokens[1]=32; ck(near(score(j).missing,0),"exclude unavailable history");
    j.live={0,0,0}; ck(score(j).distributions==0,"empty candidate set is not a zero-risk sample");
    j=job(); j.layers={3,63}; j.queries={0x3c00,0x3c00,0xbc00,0xbc00};
    for (size_t b=0;b<3;++b) {auto k=std::make_shared<MismatchKeyBlock>();k->keys={j.blocks[b]->keys[0],j.blocks[b]->keys[0]};j.blocks[b]=k;}
    ck(near(score(j).missing,0.5*((1-1/z)+(1-1/(1+std::exp(-1.0)+std::exp(-2.0))))),"normalize heads/layers before averaging");
    // Independent dense double-precision oracle exercises the SIMD dot path,
    // grouped heads, multiple Query rows, and online-softmax reduction.
    MismatchJob dense;dense.layers={3,31,63};dense.heads=4;dense.kv_heads=2;dense.dim=256;
    dense.block_tokens=32;dense.indexed_tokens=dense.position=9*32;dense.tail_begin=9*32;
    dense.query_positions={63,127,255,287};dense.sink_blocks=1;
    dense.selected_tokens={32,0,32,0,32,0,32,0,32};dense.live.assign(9,1);
    dense.mandatory_tokens={16,0,8,0,32,0,0,0,16};
    for (uint32_t b=0;b<9;++b) {
        auto k=std::make_shared<MismatchKeyBlock>();k->keys.resize(3*2*256);
        for (size_t i=0;i<k->keys.size();++i) k->keys[i]=(i+b)%2?0xb400+(i*17+b*31)%1024:0x3400+(i*13+b*7)%1024;
        dense.blocks.push_back(k);
    }
    dense.queries.resize(3*4*4*256);
    for (size_t i=0;i<dense.queries.size();++i) dense.queries[i]=0x3000+(i*11)%1024;
    double oracle=0, oracle_cm[4]={0,0,0,0};
    for (size_t l=0;l<3;++l) for (size_t r=0;r<4;++r) for (size_t h=0;h<4;++h) {
        double z=0, outside=0, mandatory=0;
        for (size_t b=0;b<(dense.query_positions[r]+1)/32;++b) {
            double dot=0;
            for (size_t d=0;d<256;++d) dot+=double(mismatch_half_to_float(dense.queries[((l*4+r)*4+h)*256+d]))*
                mismatch_half_to_float(dense.blocks[b]->keys[(l*2+h/2)*256+d]);
            const double w=std::exp(dot/16);z+=w;if (!dense.selected_tokens[b]) outside+=w;
            mandatory+=w*dense.mandatory_tokens[b]/32;
        }
        oracle+=outside/z;
        oracle_cm[r]+=mandatory/z/12;
    }
    const auto dense_score=score(dense);
    ck(near(dense_score.missing,oracle/48,1e-6),"SIMD online mass differs from dense FP64 oracle");
    for (size_t r=0;r<4;++r) {
        const uint64_t n=dense.query_positions[r]+1;
        uint64_t b=0,m=0;
        for (size_t block=0;block<n/32;++block) {
            b+=dense.selected_tokens[block];m+=dense.mandatory_tokens[block];
        }
        const double expected=oracle_cm[r]+double(b-m)/(n-m)*(1-oracle_cm[r]);
        ck(near(dense_score.query_mandatory_coverage[r],oracle_cm[r],1e-6) &&
           near(dense_score.query_random_coverage[r],expected,1e-6) &&
           near(dense_score.query_lift[r],dense_score.query_coverage[r]/expected,1e-6),
           "mandatory mass and reference normalize heads/layers before forming lift");
    }
    // The completed decode tail can lag the official Mean-K index. Its raw
    // immutable K supplement must yield the same distribution, for each dtype.
    for (unsigned bytes:{1u,2u,4u}) {
        j=job();
        auto raw=std::make_shared<MismatchKeyBlock>();raw->raw_tokens=32;raw->raw_elem_bytes=bytes;
        raw->raw.resize(32*bytes);
        for (size_t i=0;i<32;++i) {
            if (bytes==1) raw->raw[i]=0x40; // E4M3 2.0
            if (bytes==2) {uint16_t v=0x4000;std::memcpy(raw->raw.data()+i*2,&v,2);}
            if (bytes==4) {float v=2;std::memcpy(raw->raw.data()+i*4,&v,4);}
        }
        j.blocks[2]=raw;
        ck(near(score(j).missing,1-1/z),"raw tail supplement dtype normalization");
    }
    MismatchQueryMean mean;
    const uint16_t qa[]={0x3c00,0x4000,0x4200,0x4400, 0xbc00,0xc000,0xc200,0xc400};
    auto m=mean.append(qa,2,4,2,1,0,4);
    ck(m.empty() && mean.count==2,"accepted prefix retained across speculative commit");
    m=mean.append(qa,2,4,4,1,0,4,2);
    ck(m.size()==1 && m[0].position==3 && m[0].tokens==4 &&
       near(m[0].values[0],2.5) && near(m[0].values[1],-2.5),"mean query across layers and accepted rows");
    mean.append(qa,2,4,1,1,4,4);
    m=mean.append(qa,2,4,2,1,10,4);
    ck(mean.discarded_tokens==1 && m.size()==1 && m[0].tokens==2,"discontinuous context must not contaminate mean");
    ActionAttentionParser parser;parser.scope=ActionAttentionParser::Scope::AllOutput;
    parser.think_open=10;parser.think_close=11;parser.interval=1;
    const uint32_t thought[]={10,12,13,11,14};parser.prepare(thought,5);
    ck(parser.commit(3)==7 && parser.committed.tokens==3,"thinking accepted prefix sampled");
    ck(parser.commit(5)==24 && parser.committed.tokens==5 && parser.commit(5)==0,"all output commit idempotent");
    const double d=mismatch_decay(0,.8,.1,1024,8192);
    ck(d>0 && mismatch_decay(d,.8,.1,1024,8192)>d,"persistent omission raises debt");
    ck(mismatch_decay(d,0,.1,1024,8192)<d,"low omission lowers debt");
    ck(near(mismatch_decay(d,.8,.1,1024,8192),mismatch_decay(mismatch_decay(d,.8,.1,512,8192),.8,.1,512,8192)),"token-based decay independent of probe density");
    ck(mismatch_decay(0,0,.1,1024,8192)==0,"debt cannot become negative");
    // A submitted immutable snapshot survives replacement of producer blocks.
    auto frozen=job(); auto producer=frozen.blocks;
    auto replacement=std::make_shared<MismatchKeyBlock>();replacement->keys={0x7c00};producer[0]=replacement;
    ck(near(score(frozen).missing,1-1/z),"immutable index snapshot");
    const auto observe=[](const MismatchJob &snapshot) {
        const auto path="/tmp/qw3-mismatch-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".jsonl";
        qw3::KvMemMismatchConfig cfg;cfg.log_path=path;
        { MismatchObserver worker(cfg); ck(worker.submit(snapshot),"async submission");worker.finish();ck(!worker.busy(),"drain at teardown"); }
        std::ifstream file(path);std::string line;std::getline(file,line);std::remove(path.c_str());
        return line;
    };
    std::string line=observe(frozen);
    ck(line.find("\"status\":\"ok\"")!=std::string::npos && line.find("\"refresh_triggered\":false")!=std::string::npos &&
       line.find("\"sink_excluded\":false")!=std::string::npos && line.find("\"query_coverage\":[")!=std::string::npos,"observation log");
    ck(line.find("\"schema_version\":3")!=std::string::npos &&
       line.find("\"baseline_selector_distribution\":false")!=std::string::npos &&
       line.find("\"query_candidate_tokens\":[96]")!=std::string::npos &&
       line.find("\"query_baseline_status\":[\"ok\"]")!=std::string::npos &&
       line.find("\"query_baseline_valid\":[true]")!=std::string::npos,"normalized schema metadata and aligned arrays");
    j=job();j.selected_tokens={0,0,0};line=observe(j);
    ck(line.find("\"query_random_coverage\":[0]")!=std::string::npos &&
       line.find("\"query_lift\":[null]")!=std::string::npos &&
       line.find("\"query_baseline_status\":[\"zero_expectation\"]")!=std::string::npos &&
       line.find("\"random_coverage\":0,\"lift\":null")!=std::string::npos,"zero expectation serializes undefined lift as null");
    j.live={0,0,0};line=observe(j);
    ck(line.find("\"query_coverage\":[null]")!=std::string::npos &&
       line.find("\"query_mandatory_coverage\":[null]")!=std::string::npos &&
       line.find("\"query_baseline_valid\":[false]")!=std::string::npos &&
       line.find("\"random_coverage\":null,\"lift\":null")!=std::string::npos,"empty candidates preserve null query identity");
    j=job();j.mandatory_tokens={0,1,0};line=observe(j);
    ck(line.find("\"status\":\"error\"")!=std::string::npos,"invalid mandatory snapshot fails only its observer job");
    std::cout<<"PASS CPU mismatch: causal/GQA/layer normalization, mandatory token reference, exact random enumeration, partial visibility, sink/live masks, immutable snapshots, token decay and async logging\n";
    return 0;
} catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
