#include "host_io_buffer_pool.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>

static void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct PinnedBuffer : qw3::HostBuffer {
    explicit PinnedBuffer(uint64_t n) {
        bytes = n; pinned = true; check(cudaHostAlloc(&data, n, cudaHostAllocDefault));
    }
    ~PinnedBuffer() override { if (data) cudaFreeHost(data); }
};
__global__ void delay_copy() {
    const auto start = clock64();
    while (clock64() - start < 1000000) {}
}
int main() try {
    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    unsigned char *device; check(cudaMalloc(&device, 4096));
    {
        qw3::detail::HostIoBufferPool pool(
            [](uint64_t n, const char *) { return std::make_unique<PinnedBuffer>(n); },
            8192, [&] { return cudaStreamSynchronize(stream) == cudaSuccess; });
        unsigned char received[4096];
        for (unsigned value = 1; value <= 128; ++value) {
            const void *previous = nullptr;
            try {
                auto source = pool.acquire(4096, "pending copy");
                previous = source->data; std::memset(source->data, value, 4096);
                delay_copy<<<1,1,0,stream>>>(); check(cudaGetLastError());
                check(cudaMemcpyAsync(device, source->data, 4096, cudaMemcpyHostToDevice, stream));
                if (value % 2) throw value; // Exercise unwind before H2D completes.
            } catch (unsigned) {}
            auto next = pool.acquire(4096, "next request");
            if (next->data != previous) throw std::runtime_error("backing storage was not reused");
            std::memset(next->data, 0, 4096);
            check(cudaMemcpy(received, device, 4096, cudaMemcpyDeviceToHost));
            for (unsigned char actual : received)
                if (actual != value) throw std::runtime_error("reused pinned source before H2D completed");
        }
        const auto stats = pool.stats();
        if (stats.allocations != 1 || stats.reuses != 255 || stats.cached_bytes != 4096)
            throw std::runtime_error("unexpected CUDA pool accounting");
    }
    check(cudaFree(device)); check(cudaStreamDestroy(stream));
    std::cout << "pooled_io_cuda asynchronous_copy_and_exception_cases=128 status=pass\n";
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
