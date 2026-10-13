#include "dflash_weights.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace qw3 {

DFlashWeights::DFlashWeights(const DFlashModel &model, DeviceBackend &backend)
    : config_(model.config()), identity_sha256_(model.identity_sha256()) {
    for (const auto &[name, tensor] : model.tensors()) {
        const uint64_t cols = tensor->dims.front();
        uint64_t rows = 1;
        for (size_t i = 1; i < tensor->dims.size(); ++i) rows *= tensor->dims[i];
        // Norms and convolution base coefficients are consumed by FP32
        // elementwise kernels. Do not apply Qwen3.5's +1 norm transform.
        const bool f32 = tensor->type == ModelTensorType::F32 || tensor->dims.size() == 1 ||
                         name.find(".base_kernel") != std::string::npos;
        std::unique_ptr<DeviceWeight> weight;
        if (f32) {
            std::vector<float> staging;
            const float *values = static_cast<const float *>(tensor->data);
            if (tensor->type == ModelTensorType::BF16) {
                staging.resize(rows * cols);
                const auto *bf16 = static_cast<const uint16_t *>(tensor->data);
                for (size_t i = 0; i < staging.size(); ++i) {
                    const uint32_t bits = uint32_t(bf16[i]) << 16;
                    std::memcpy(&staging[i], &bits, sizeof(bits));
                }
                values = staging.data();
            }
            weight = backend.weight_f32(values, rows * cols, name.c_str());
            weight->rows = rows;
            weight->cols = cols;
            bytes_ += rows * cols * sizeof(float);
        } else {
            weight = backend.weight_bf16(tensor->data, rows, cols, name.c_str());
            bytes_ += rows * cols * sizeof(uint16_t);
        }
        weights_.emplace(name, std::move(weight));
    }
}

DeviceWeight &DFlashWeights::at(const std::string &name) const {
    const auto it = weights_.find(name);
    if (it == weights_.end()) throw std::runtime_error("unknown DFlash device weight: " + name);
    return *it->second;
}

} // namespace qw3
