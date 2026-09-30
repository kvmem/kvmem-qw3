#pragma once
#include "qw3/target_sampling.hpp"
#include <cuda_runtime.h>
#include <memory>

namespace qw3 {
class TargetSamplerCuda {
public:
    TargetSamplerCuda();
    ~TargetSamplerCuda();
    cudaError_t run(const float *logits, const DeviceSamplingRequest &request,
                    DeviceSamplingResult &result, cudaStream_t stream);
private:
    struct Storage;
    std::unique_ptr<Storage> storage_;
};
} // namespace qw3
