// Candidate screening only: keep decode's Q8_1 quantizer and repack its exact
// values/scales for the existing INT8 MMA kernel. No runtime dispatch changes.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace qw3::ported {
bool launch_quantize_q8_1(const float *, void *, uint32_t, uint32_t, uint32_t, cudaStream_t);
bool launch_mmvq_q8_0(const uint8_t *, const void *, float *, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t);
bool launch_mmq_q8_0_medium(const uint8_t *, const void *, float *, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t);
}
namespace {
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool b, const char *message) { if (!b) throw std::runtime_error(message); }
struct alignas(4) DecodeBlock { half2 ds; int8_t qs[32]; };
struct alignas(16) MmaBlock { float d[4]; int8_t qs[128]; };
static_assert(sizeof(DecodeBlock)==36 && sizeof(MmaBlock)==144);
template<class T> struct Buffer {
    T *p=nullptr; size_t n;
    explicit Buffer(size_t size):n(size) { check(cudaMalloc(&p,n*sizeof(T))); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer &)=delete;
    void upload(const std::vector<T> &v) { require(v.size()==n,"upload size");check(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice)); }
    std::vector<T> read() { std::vector<T> v(n);check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v; }
};
__global__ void repack(const DecodeBlock *src, MmaBlock *dst, unsigned batch, unsigned blocks) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=batch*blocks*32) return;
    const unsigned q=i%32, k=(i/32)%blocks, row=i/(blocks*32);
    const auto &from=src[row*blocks+k];
    auto &to=dst[(k/4)*batch+row];
    to.qs[(k%4)*32+q]=from.qs[q];
    if (!q) to.d[k%4]=__low2float(from.ds);
}
void pack(const DecodeBlock *src, MmaBlock *dst, unsigned batch, unsigned cols) {
    repack<<<(batch*cols+255)/256,256>>>(src,dst,batch,cols/32);
    check(cudaGetLastError());
}
template<class F> float time_ms(F f) {
    for (int i=0;i<3;++i) f();
    cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));
    check(cudaEventRecord(a));for (int i=0;i<20;++i)f();check(cudaEventRecord(b));check(cudaEventSynchronize(b));
    float ms=0;check(cudaEventElapsedTime(&ms,a,b));check(cudaEventDestroy(a));check(cudaEventDestroy(b));return ms/20;
}
void shape(unsigned rows, unsigned cols, bool timing) {
    require(cols%256==0,"prototype requires aligned Q8 weight planes");
    constexpr unsigned max_batch=64;constexpr float guard=-1024;
    const unsigned blocks=cols/32,stride=rows+7;
    std::mt19937 rng(781+cols+rows);
    std::uniform_int_distribution<int> quants(-127,127);
    std::uniform_real_distribution<float> floats(-1,1);
    std::vector<uint8_t> weights(size_t(rows)*blocks*34);
    for (unsigned row=0;row<rows;++row) {
        auto *p=weights.data()+size_t(row)*blocks*34;
        for (unsigned k=0;k<blocks;++k) {
            const half scale=__float2half(.0007f+.00001f*(k%17));
            std::memcpy(p+2*k,&scale,2);
            for (unsigned q=0;q<32;++q) p[2*blocks+32*k+q]=uint8_t(int8_t(quants(rng)));
        }
    }
    std::vector<float> input(size_t(max_batch)*cols);
    for (auto &x:input)x=floats(rng);
    // Include a zero row and small values to exercise zero and half-rounded scales.
    std::fill_n(input.data(),cols,0.f);
    for (unsigned k=0;k<cols;++k)input[cols+k]*=.001f;
    Buffer<uint8_t> w(weights.size());w.upload(weights);
    Buffer<float> x(input.size());x.upload(input);
    Buffer<DecodeBlock> q(size_t(max_batch)*blocks);
    require(qw3::ported::launch_quantize_q8_1(x.p,q.p,max_batch,cols,cols,nullptr),"quantize launch");
    const auto quantized=q.read();
    Buffer<MmaBlock> packed(size_t(max_batch)*blocks/4);
    Buffer<float> reference(size_t(max_batch+1)*stride),candidate(reference.n),single(stride+7);
    const std::vector<float> guards(reference.n,guard);
    // Batch=1 MMA is the arithmetic oracle for batch membership/width changes.
    std::vector<float> independent(size_t(max_batch)*rows);
    for (unsigned b=0;b<max_batch;++b) {
        pack(q.p+size_t(b)*blocks,packed.p,1,cols);
        require(qw3::ported::launch_mmq_q8_0_medium(w.p,packed.p,single.p,rows,cols,1,stride,nullptr),"MMA single launch");
        const auto actual=single.read();std::copy_n(actual.data(),rows,independent.data()+size_t(b)*rows);
    }
    for (unsigned batch:{1u,8u,9u,16u,17u,24u,32u,33u,64u}) {
        auto old=[&] {
            for (unsigned b=0;b<batch;b+=8)
                require(qw3::ported::launch_mmvq_q8_0(w.p,q.p+size_t(b)*blocks,reference.p+size_t(b)*stride,
                        rows,cols,std::min(8u,batch-b),stride,nullptr),"MMVQ launch");
        };
        auto mma=[&] {
            pack(q.p,packed.p,batch,cols);
            require(qw3::ported::launch_mmq_q8_0_medium(w.p,packed.p,candidate.p,rows,cols,batch,stride,nullptr),"MMA launch");
        };
        reference.upload(guards);candidate.upload(guards);old();mma();
        const auto a=reference.read(),b=candidate.read();
        const auto p=packed.read();
        for (unsigned r=0;r<batch;++r)for (unsigned k=0;k<blocks;++k) {
            const auto &from=quantized[size_t(r)*blocks+k];const auto &to=p[(k/4)*batch+r];
            require(to.d[k%4]==__half2float(__low2half(from.ds)),"changed activation scale");
            require(!std::memcmp(to.qs+(k%4)*32,from.qs,32),"changed activation integers");
        }
        float max_error=0;double sum_error=0,sum_ref=0;
        for (unsigned r=0;r<=max_batch;++r)for(unsigned j=0;j<stride;++j) {
            const size_t i=size_t(r)*stride+j;
            if(r>=batch||j>=rows) { require(a[i]==guard&&b[i]==guard,"output guard overwritten");continue; }
            require(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite result");
            require(!std::memcmp(&b[i],&independent[size_t(r)*rows+j],sizeof(float)),"MMA depends on batch width");
            const float error=std::abs(a[i]-b[i]);max_error=std::max(error,max_error);
            require(error<=1e-4f+1e-5f*std::abs(a[i]),"MMA exceeds fixed operator error tolerance");
            sum_error+=double(error)*error;sum_ref+=double(a[i])*a[i];
        }
        std::printf("MMA_VERIFY rows=%u cols=%u batch=%u quantized_input_exact=1 batch_bit_exact=1 guards=ok max_abs=%.9g rel_l2=%.9g\n",
                    rows,cols,batch,max_error,std::sqrt(sum_error/std::max(1e-30,sum_ref)));
        if(timing&&(batch==8||batch==16||batch==24||batch==32))for(unsigned trial=0;trial<3;++trial) {
            float old_ms,mma_ms;
            if(trial%2){mma_ms=time_ms(mma);old_ms=time_ms(old);}else{old_ms=time_ms(old);mma_ms=time_ms(mma);}
            std::printf("MMA_TIMING rows=%u cols=%u batch=%u trial=%u old_ms=%.6f mma_ms=%.6f ratio=%.4f repack_included=1 common_quantizer_excluded=1\n",
                        rows,cols,batch,trial,old_ms,mma_ms,old_ms/mma_ms);
        }
        std::fflush(stdout);
    }
}
}
int main() try {
    shape(65,256,false);shape(5120,6144,true);shape(17408,5120,true);shape(5120,17408,true);
    std::puts("MMA_SCREENING status=passed runtime_enabled=0 model_quality_qualified=0");return 0;
} catch(const std::exception &e) {std::fprintf(stderr,"MMA_SCREENING failed: %s\n",e.what());return 1;}
