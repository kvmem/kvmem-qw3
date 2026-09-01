#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qw3::detail {

struct CpuVisionImage {
    std::string media_type;
    std::string base64_data;
};

struct CpuVisionEncoding {
    struct Grid {
        uint32_t temporal = 0;
        uint32_t height = 0;
        uint32_t width = 0;
        uint32_t rows = 0;
    };

    uint32_t embedding_dim = 0;
    bool cache_hit = false;
    std::vector<Grid> grids;
    // Image-major contiguous float32 rows, sum(grid.rows) * embedding_dim.
    std::vector<float> embeddings;
};

class CpuVisionFrontend {
public:
    CpuVisionFrontend(std::string model_directory,
                      std::string python_executable,
                      std::string worker_script,
                      uint32_t threads);
    ~CpuVisionFrontend();

    CpuVisionFrontend(const CpuVisionFrontend &) = delete;
    CpuVisionFrontend &operator=(const CpuVisionFrontend &) = delete;

    CpuVisionEncoding encode(const std::vector<CpuVisionImage> &images);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace qw3::detail
