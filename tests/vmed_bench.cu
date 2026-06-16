// Standalone microbench + parity check for the medium-batch vmed MMA kernel.
//   - parity: vmed output vs mmvq output at several M (both read identical
//     split-plane Q8 weights; activations quantized by each path's own
//     quantizer, so agreement to ~1e-2 relative validates the MMA core).
//   - scaling: time mmvq (M<=8), vmed, mmq across M to find the crossover and
//     confirm vmed beats mmq's flat tile floor at medium M.
//
// Build:
//   nvcc -O3 -arch=sm_120a -std=c++17 tests/vmed_bench.cu \
//     build/CMakeFiles/qw3_core.dir/src/{mmvq_q8,mmq_q8}.cu.o -o /tmp/vmed_bench
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace qw3 { namespace ported {
size_t q8_1_scratch_bytes(uint32_t batch, uint32_t cols);
size_t q8_1_mmq_scratch_bytes(uint32_t batch, uint32_t cols);
bool launch_quantize_q8_1(const float*, void*, uint32_t, uint32_t, uint32_t, cudaStream_t);
bool launch_quantize_mmq_q8_1(const float*, void*, uint32_t, uint32_t, uint32_t, cudaStream_t);
bool launch_mmvq_q8_0(const uint8_t*, const void*, float*, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t);
bool launch_mmq_q8_0(const uint8_t*, const void*, float*, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t);
bool launch_mmq_q8_0_medium(const uint8_t*, const void*, float*, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t);
}}

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

using qw3::ported::q8_1_scratch_bytes;
using qw3::ported::q8_1_mmq_scratch_bytes;
using qw3::ported::launch_quantize_q8_1;
using qw3::ported::launch_quantize_mmq_q8_1;
using qw3::ported::launch_mmvq_q8_0;
using qw3::ported::launch_mmq_q8_0;
using qw3::ported::launch_mmq_q8_0_medium;

int main() {
    const uint32_t rows = 4096, cols = 4096;
    const uint32_t n_blocks = cols / 32;              // 128
    const size_t   wbytes   = (size_t)rows * n_blocks * 34;

    // ---- build split-plane Q8 weight: [fp16 scales | int8 quants] per row ----
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> sd(0.005f, 0.02f);
    std::vector<uint8_t> hW(wbytes);
    for (uint32_t r = 0; r < rows; ++r) {
        uint8_t *row = hW.data() + (size_t)r * n_blocks * 34;
        __half *S = reinterpret_cast<__half*>(row);
        int8_t *Q = reinterpret_cast<int8_t*>(row + 2 * n_blocks);
        for (uint32_t b = 0; b < n_blocks; ++b) S[b] = __float2half(sd(rng));
        for (uint32_t i = 0; i < n_blocks * 32; ++i) Q[i] = (int8_t)qd(rng);
    }
    uint8_t *dW; CK(cudaMalloc(&dW, wbytes));
    CK(cudaMemcpy(dW, hW.data(), wbytes, cudaMemcpyHostToDevice));

    const uint32_t MAXB = 256;
    std::vector<float> hX((size_t)MAXB * cols);
    std::uniform_real_distribution<float> xd(-1.f, 1.f);
    for (auto &v : hX) v = xd(rng);
    float *dX; CK(cudaMalloc(&dX, hX.size()*sizeof(float)));
    CK(cudaMemcpy(dX, hX.data(), hX.size()*sizeof(float), cudaMemcpyHostToDevice));

    void *dQ1, *dQmmq; float *dDvmvq, *dDmmq, *dDvmed;
    CK(cudaMalloc(&dQ1,   q8_1_scratch_bytes(MAXB, cols)));
    CK(cudaMalloc(&dQmmq, q8_1_mmq_scratch_bytes(MAXB, cols)));
    CK(cudaMalloc(&dDvmvq, (size_t)MAXB*rows*sizeof(float)));
    CK(cudaMalloc(&dDmmq,  (size_t)MAXB*rows*sizeof(float)));
    CK(cudaMalloc(&dDvmed, (size_t)MAXB*rows*sizeof(float)));
    cudaStream_t st; CK(cudaStreamCreate(&st));

    auto stage = [&](uint32_t B){
        launch_quantize_q8_1(dX, dQ1, B, cols, cols, st);
        launch_quantize_mmq_q8_1(dX, dQmmq, B, cols, cols, st);
        CK(cudaStreamSynchronize(st));
    };

    // ---------------- parity: vmed vs mmvq ----------------
    printf("=== parity (vmed vs mmvq, max|rel| over first 4096 outputs) ===\n");
    for (uint32_t B : {8u, 16u, 24u, 32u, 48u, 64u}) {
        stage(B);
        std::vector<float> a((size_t)B*rows), b((size_t)B*rows);
        if (B <= 8) {
            launch_mmvq_q8_0(dW, dQ1, dDvmvq, rows, cols, B, rows, st);
            CK(cudaStreamSynchronize(st));
            CK(cudaMemcpy(a.data(), dDvmvq, a.size()*sizeof(float), cudaMemcpyDeviceToHost));
        } else {
            // reference: mmq (v8) which is the established large-batch MMA path
            launch_mmq_q8_0(dW, dQmmq, dDmmq, rows, cols, B, rows, st);
            CK(cudaStreamSynchronize(st));
            CK(cudaMemcpy(a.data(), dDmmq, a.size()*sizeof(float), cudaMemcpyDeviceToHost));
        }
        bool ok = launch_mmq_q8_0_medium(dW, dQmmq, dDvmed, rows, cols, B, rows, st);
        CK(cudaStreamSynchronize(st));
        if (!ok) { printf("  B=%-3u vmed REJECTED launch\n", B); continue; }
        CK(cudaMemcpy(b.data(), dDvmed, b.size()*sizeof(float), cudaMemcpyDeviceToHost));
        double maxrel = 0; int n = std::min((size_t)4096, a.size());
        for (int i = 0; i < n; ++i) {
            double d = fabs(a[i]-b[i]); double den = fabs(a[i])+1e-6;
            maxrel = std::max(maxrel, d/den);
        }
        printf("  B=%-3u maxrel=%.4e  ref0=%.4f vmed0=%.4f  %s\n",
               B, maxrel, a[0], b[0], maxrel<2e-2?"OK":"MISMATCH");
    }

    // ---------------- scaling timing ----------------
    auto timeit = [&](auto fn, int iters)->float{
        cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int w=0;w<5;++w) fn();
        CK(cudaStreamSynchronize(st));
        CK(cudaEventRecord(e0,st));
        for (int i=0;i<iters;++i) fn();
        CK(cudaEventRecord(e1,st)); CK(cudaEventSynchronize(e1));
        float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1));
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        return ms/iters;
    };

    printf("\n=== scaling (ms/call; iters=200) ===\n");
    printf("%4s %10s %10s %10s\n","M","mmvq","vmed","mmq");
    for (uint32_t B : {1u,2u,4u,8u,16u,24u,32u,48u,64u,96u,128u,192u,256u}) {
        stage(B);
        float tv = (B<=8) ? timeit([&]{ launch_mmvq_q8_0(dW,dQ1,dDvmvq,rows,cols,B,rows,st); },200) : -1;
        float tm = (B<=64)? timeit([&]{ launch_mmq_q8_0_medium(dW,dQmmq,dDvmed,rows,cols,B,rows,st);},200) : -1;
        float tq =          timeit([&]{ launch_mmq_q8_0(dW,dQmmq,dDmmq,rows,cols,B,rows,st); },200);
        printf("%4u %10.4f %10.4f %10.4f\n", B,
               tv<0?0.f:tv, tm<0?0.f:tm, tq);
    }
    printf("DONE\n");
    return 0;
}
