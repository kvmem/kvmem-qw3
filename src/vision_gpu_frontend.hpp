#pragma once

#include "qw3/device_backend.hpp"
#include "vision_cpu_frontend.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qw3::detail {

// Native BF16 Qwen3.5 visual tower. Image decoding/resizing/patchification is
// delegated to the lightweight Python preprocessor, while every learned
// operation and all projected embeddings remain on the CUDA device.
class GpuVisionFrontend {
public:
    GpuVisionFrontend(DeviceBackend &backend,
                      std::string model_directory,
                      std::string python_executable,
                      std::string worker_script,
                      uint32_t preprocess_threads);
    ~GpuVisionFrontend();

    GpuVisionFrontend(const GpuVisionFrontend &) = delete;
    GpuVisionFrontend &operator=(const GpuVisionFrontend &) = delete;

    CpuVisionEncoding encode(const std::vector<CpuVisionImage> &images);
    uint64_t weight_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace qw3::detail
