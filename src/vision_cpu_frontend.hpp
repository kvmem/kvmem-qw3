#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "qw3/qw3.hpp"

namespace qw3::detail {

using CpuVisionImage = VisionImage;
using CpuVisionEncoding = VisionEncoding;

struct CpuVisionPatches {
    using Grid = CpuVisionEncoding::Grid;
    uint32_t patch_dim = 0;
    std::vector<Grid> grids;
    // Image-major flattened patches in native BF16 bit representation.
    std::vector<uint16_t> patches;
};

class CpuVisionFrontend {
public:
    CpuVisionFrontend(std::string model_directory,
                      std::string python_executable,
                      std::string worker_script,
                      uint32_t threads,
                      bool preprocess_only = false,
                      std::string runtime_device = "cpu");
    ~CpuVisionFrontend();

    CpuVisionFrontend(const CpuVisionFrontend &) = delete;
    CpuVisionFrontend &operator=(const CpuVisionFrontend &) = delete;

    CpuVisionEncoding encode(const std::vector<CpuVisionImage> &images);
    CpuVisionPatches preprocess(const std::vector<CpuVisionImage> &images);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace qw3::detail
