#include "qw3/device_backend.hpp"
#include "vision_gpu_frontend.hpp"
#include "vision_embedding_storage.hpp"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

using namespace qw3;
using namespace qw3::detail;
static void check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static VisionImage load(const char *path) {
    std::ifstream f(path, std::ios::binary);
    check(bool(f), "missing image");
    std::vector<unsigned char> bytes(std::istreambuf_iterator<char>(f), {});
    static constexpr char abc[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string b64;
    for (size_t i = 0; i < bytes.size(); i += 3) {
        unsigned x = unsigned(bytes[i]) << 16;
        if (i+1 < bytes.size()) x |= unsigned(bytes[i+1]) << 8;
        if (i+2 < bytes.size()) x |= bytes[i+2];
        b64 += abc[(x>>18)&63]; b64 += abc[(x>>12)&63];
        b64 += i+1 < bytes.size() ? abc[(x>>6)&63] : '=';
        b64 += i+2 < bytes.size() ? abc[x&63] : '=';
    }
    return {"image/png", b64};
}
static std::vector<uint16_t> read(DeviceBackend &backend, const VisionEncoding &result) {
    auto *s = dynamic_cast<const DeviceInputEmbeddingStorage *>(result.storage.get());
    check(s != nullptr, "expected device embeddings");
    std::vector<uint16_t> bytes(uint64_t(s->rows())*s->dim());
    for (const auto &part : s->segments()) {
        check(backend.copy_bytes_to_host(*s->resolve(part.logical_row).tensor,
            bytes.data()+uint64_t(part.logical_row)*s->dim(),
            uint64_t(part.source_row)*s->dim()*2,
            uint64_t(part.rows)*s->dim()*2).ok, "read device bytes");
    }
    return bytes;
}
int main(int argc, char **argv) {
    if (argc != 8) {
        std::cerr << "usage: test MODEL_DIR PYTHON WORKER IMAGE1 IMAGE2 IMAGE3 IMAGE4\n";
        return 2;
    }
    // Each real OSWorld screenshot is about 20 MiB of projected BF16 features.
    // The budget can keep one image, forcing actual GPU->CPU->GPU transfers.
    setenv("QW3_VISION_GPU_CACHE_MIB", "24", 1);
    setenv("QW3_VISION_CPU_CACHE_MIB", "96", 1);
    auto backend = make_cuda_device_backend(LinearBackend::Cublas);
    check(backend && backend->begin().ok, "CUDA start");
    {
        GpuVisionFrontend gpu(*backend, argv[1], argv[2], argv[3], 8);
        std::vector<VisionImage> images;
        std::vector<std::vector<uint16_t>> reference;
        for (int i = 4; i < 8; ++i) {
            images.push_back(load(argv[i]));
            auto result = gpu.encode({images.back()});
            check(result.cache_misses == 1, "test images must be distinct");
            reference.push_back(read(*backend, result));
            check(reference.back().size()*2 <= (24ULL<<20), "image too large for test GPU cache");
            check(reference.back().size()*2 > (12ULL<<20), "image too small to force eviction");
        }
        // Reorder old images and duplicate a CPU hit; identity must follow the
        // image, not its history position. All restored bytes must be exact.
        const std::vector<size_t> order = {2, 0, 3, 1, 0};
        std::vector<VisionImage> request;
        std::vector<VisionImageSourcePtr> sources;
        for (const auto &image : images) sources.push_back(std::make_shared<const VisionImageSource>(image));
        std::vector<VisionImageSourcePtr> prepared_request;
        std::vector<uint16_t> expected;
        for (auto i : order) {
            request.push_back(images[i]);
            prepared_request.push_back(sources[i]);
            expected.insert(expected.end(), reference[i].begin(), reference[i].end());
        }
        for (int round = 0; round < 3; ++round) {
            auto started = std::chrono::steady_clock::now();
            auto result = round == 0 ? gpu.encode(request) : gpu.encode_sources(prepared_request);
            const double ms = std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-started).count();
            check(result.cache_misses == 0 && result.cache_hits == request.size(),
                  "CPU history unexpectedly re-encoded");
            check(read(*backend, result) == expected, "restored BF16 features changed");
            std::cout << "restored round=" << round << " ms=" << ms
                      << " hits=" << result.cache_hits << " exact_bf16=1\n";
        }
    }
    // An undersized CPU tier must evict safely and re-encode the exact image.
    setenv("QW3_VISION_CPU_CACHE_MIB", "1", 1);
    {
        GpuVisionFrontend gpu(*backend, argv[1], argv[2], argv[3], 8);
        auto first = gpu.encode({load(argv[4])});
        auto expected = read(*backend, first);
        first = {};
        gpu.encode({load(argv[5])});
        auto again = gpu.encode({load(argv[4])});
        check(again.cache_misses == 1, "oversized host entry was retained");
        check(read(*backend, again) == expected, "eviction fallback changed BF16");
    }
    check(backend->end().ok, "CUDA end");
    std::cout << "CPU vision cache: eviction, reordered/duplicate hits, exact BF16, capacity fallback passed\n";
}
