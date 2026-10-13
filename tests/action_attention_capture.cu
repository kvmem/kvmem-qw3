#include "qw3/device_backend.hpp"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
using namespace qw3;
void ck(DeviceStatus s) {if(!s.ok) throw std::runtime_error(s.message);}
void require(bool x,const char*s) {if(!x) throw std::runtime_error(s);}
template<class KV> void run(DeviceBackend &d,bool fp8,bool prefill,unsigned base,unsigned batch,bool bench,unsigned first_row=0,unsigned row_step=1,bool dense=false) {
    const unsigned heads=24,kh=4,dim=256,page=32,len=base+batch,np=(len+31)/32,qs=heads*2*dim,os=heads*dim;
    const unsigned nb=base/32;
    std::vector<int32_t> pages(np),starts(nb),counts(nb,32);
    for(unsigned i=0;i<np;++i) pages[i]=np-i-1;
    for(unsigned i=0;i<nb;++i) starts[i]=i*32;
    std::vector<KV> k(size_t(np)*page*kh*dim),v(k.size());
    std::vector<float> q(batch*qs);
    for(size_t i=0;i<k.size();++i) {k[i]=KV(.1f*std::sin(i*.001f));v[i]=KV(.2f*std::cos(i*.002f));}
    for(size_t i=0;i<q.size();++i) q[i]=std::sin(i*.031f);
    auto kd=fp8?d.tensor_fp8_kv(k.size(),"k"):d.tensor_f16(k.size(),"k");
    auto vd=fp8?d.tensor_fp8_kv(v.size(),"v"):d.tensor_f16(v.size(),"v");
    auto qd=d.tensor_f32(q.size(),"q"),out=d.tensor_f32(batch*os,"out");
    auto pd=d.tensor_i32(np,"pages"),sd=d.tensor_i32(nb,"starts"),cd=d.tensor_i32(nb,"counts");
    ck(d.copy_bytes_from_host(*kd,0,k.data(),k.size()*sizeof(KV)));
    ck(d.copy_bytes_from_host(*vd,0,v.data(),v.size()*sizeof(KV)));
    ck(d.copy_bytes_from_host(*qd,0,q.data(),q.size()*sizeof(float)));
    ck(d.copy_i32_from_host(*pd,0,pages.data(),np));
    ck(d.copy_i32_from_host(*sd,0,starts.data(),nb));ck(d.copy_i32_from_host(*cd,0,counts.data(),nb));
    const unsigned rows=1+(batch-first_row-1)/row_step;
    require(rows && rows<=64,"invalid capture test mask range");
    const uint64_t mask=dense?(rows==64?~uint64_t(0):(1ull<<rows)-1):rows==1?1:bench?(1ull<<(rows-1)):((1ull<<1)|(1ull<<(rows-1)));
    const unsigned ns=__builtin_popcountll(mask);
    auto logits=d.tensor_f32(uint64_t(ns)*heads*len,"captured"),mass=d.tensor_f32(ns*(nb+1),"mass"),oracle=d.tensor_f32(nb+1,"oracle");
    auto invoke=[&](bool capture) {
        ck(d.attention_capture_set(capture?logits.get():nullptr,capture?mask:0,len,heads,first_row,row_step));
        auto f=prefill?&DeviceBackend::attention_prefill_batch_paged_gated_device:&DeviceBackend::attention_decode_batch_paged_gated_device;
        ck((d.*f)(*out,*qd,2*dim,*kd,*vd,*pd,np,page,heads,kh,dim,base,batch,qs,os,1.f/std::sqrt(float(dim))));
        ck(d.attention_capture_set(nullptr,0,0,0));
        if(capture) ck(d.attention_capture_reduce(*mass,*logits,mask,len,heads,base+first_row,nb,*sd,*cd,1.f,false,row_step));
    };
    std::vector<float> original(batch*os),captured(batch*os),got(ns*(nb+1)),ref(nb+1);
    invoke(false);ck(d.copy_to_host(*out,original.data(),0,original.size()));
    invoke(true);ck(d.copy_to_host(*out,captured.data(),0,captured.size()));
    require(original==captured,"capture changed attention output");
    ck(d.copy_to_host(*mass,got.data(),0,got.size()));
    unsigned slot=0;double worst=0;
    for(unsigned row=first_row;row<batch;row+=row_step) if(mask>>((row-first_row)/row_step)&1) {
        auto qr=d.tensor_f32(qs,"oracle q");std::vector<float> rq(q.begin()+row*qs,q.begin()+(row+1)*qs);
        if(prefill) for(auto &x:rq) x=__half2float(__float2half(x)); // matches packed tensor-core Q
        ck(d.copy_bytes_from_host(*qr,0,rq.data(),rq.size()*sizeof(float)));
        ck(d.block_attention_mass_paged_device(*oracle,*qr,2*dim,*kd,nb,heads,kh,kh*dim,dim,*sd,*cd,*pd,np,page,base+row+1,1.f/std::sqrt(float(dim))));
        ck(d.copy_to_host(*oracle,ref.data(),0,ref.size()));double sum=0;
        for(unsigned b=0;b<=nb;++b) {sum+=got[slot*(nb+1)+b];worst=std::max(worst,std::abs(double(got[slot*(nb+1)+b])-ref[b]/heads));}
        require(std::abs(sum-1)<2e-5,"attention mass not normalized");++slot;
    }
    if(worst>=2e-5) {
        std::printf("mass mismatch fp8=%d prefill=%d batch=%u worst=%.9g\n",fp8,prefill,batch,worst);
        for(unsigned i=0;i<std::min(nb+1,10u);++i) std::printf("  b=%u got=%.8f ref=%.8f\n",i,got[(ns-1)*(nb+1)+i],ref[i]/heads);
        std::vector<float> raw(uint64_t(ns)*heads*len);ck(d.copy_to_host(*logits,raw.data(),0,raw.size()));
        const unsigned row=batch==1?0:1;
        for(unsigned t=0;t<6;++t) {
            double dot=0;for(unsigned c=0;c<dim;++c)dot+=q[row*qs+c]*float(k[(pages[t/page]*page+t%page)*kh*dim+c]);
            std::printf("  t=%u tap=%.8f dot=%.8f\n",t,raw[t],dot/std::sqrt(double(dim)));
        }
    }
    require(worst<2e-5,"captured mass disagrees with independent Q*K oracle");
    // Sequential layer reductions reuse logits and one output tensor. Unequal
    // weights and deliberately different logits test overwrite/accumulate and
    // every causal row, with the already validated single-layer distribution.
    auto mixed=d.tensor_f32(ns*(nb+1),"weighted mean");
    ck(d.attention_capture_reduce(*mixed,*logits,mask,len,heads,base+first_row,nb,*sd,*cd,.25f,false,row_step));
    ck(d.attention_capture_reduce(*mixed,*logits,mask,len,heads,base+first_row,nb,*sd,*cd,.5f,true,row_step));
    std::vector<float> raw(uint64_t(ns)*heads*len),expected(ns*(nb+1));
    ck(d.copy_to_host(*logits,raw.data(),0,raw.size()));
    for(auto &x:raw)x=-x;
    ck(d.copy_bytes_from_host(*logits,0,raw.data(),raw.size()*sizeof(float)));
    ck(d.attention_capture_reduce(*mass,*logits,mask,len,heads,base+first_row,nb,*sd,*cd,1.f,false,row_step));
    ck(d.copy_to_host(*mass,expected.data(),0,expected.size()));
    ck(d.attention_capture_reduce(*mixed,*logits,mask,len,heads,base+first_row,nb,*sd,*cd,.25f,true,row_step));
    ck(d.copy_to_host(*mixed,captured.data(),0,expected.size()));
    for(size_t i=0;i<expected.size();++i)
        require(std::abs(captured[i]-(.75f*got[i]+.25f*expected[i]))<2e-6,"weighted multi-layer accumulation mismatch");
    invoke(true); // restore the original captured logits before timing
    std::printf("PASS fp8=%d prefill=%d base=%u batch=%u first_row=%u row_step=%u samples=%u bit_exact=1 weighted_mean=1 mass_max_error=%.9g\n",fp8,prefill,base,batch,first_row,row_step,ns,worst);
    if(bench) {
        for(int i=0;i<20;++i) invoke(i%2);
        // Alternate trials. 16 tokens per sampled row, including the D2H cost
        // used at accepted commit. Include unsampled calls and their sync.
        for(int trial=0;trial<8;++trial) for(int mode=0;mode<2;++mode) {
            ck(d.synchronize());auto a=std::chrono::steady_clock::now();
            for(int i=0;i<128;++i) {
                bool sample=mode && (i%(16/batch)==0);
                invoke(sample);
                if(sample) ck(d.copy_to_host(*mass,got.data(),0,got.size()));
                else ck(d.synchronize());
            }
            ck(d.synchronize());double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-a).count()/128;
            std::printf("BENCH fp8=%d prefill=%d base=%u batch=%u trial=%d enabled=%d call_ms=%.6f\n",fp8,prefill,base,batch,trial,mode,ms);
        }
    }
}
int main(int argc,char**argv) try {
    auto d=make_cuda_device_backend(LinearBackend::Cublas);require(bool(d),"no CUDA backend");ck(d->begin());
    const bool bench=argc>1;
    for(bool fp8:{false,true}) for(bool prefill:{false,true})
        for(unsigned batch:{1u,8u}) {
            if(fp8) run<__nv_fp8_e4m3>(*d,true,prefill,bench?36864:1023,batch,bench);
            else if(!bench) run<half>(*d,false,prefill,1023,batch,false);
        }
    if(!bench) {
        run<__nv_fp8_e4m3>(*d,true,false,36860,8,false);
        run<__nv_fp8_e4m3>(*d,true,true,36860,8,false);
        // Long chunks sample their last row beyond the old 64-row mask limit.
        run<half>(*d,false,true,1023,1024,false,1023);
        run<__nv_fp8_e4m3>(*d,true,true,1023,1024,false,1023);
        run<__nv_fp8_e4m3>(*d,true,true,36864,1024,false,1023);
        // Nonzero sample origin must survive the backend's tile/partition split.
        run<__nv_fp8_e4m3>(*d,true,true,36860,8,false,7);
        run<__nv_fp8_e4m3>(*d,true,false,36860,8,false,7);
        for(unsigned step:{512u,256u,32u,16u})
            run<__nv_fp8_e4m3>(*d,true,true,36864,1024,false,step-1,step,true);
    }
    return 0;
} catch(const std::exception&e) {std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
