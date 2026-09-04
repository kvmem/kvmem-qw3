#pragma once

#include "qw3/device_backend.hpp"
#include "qw3/qw3.hpp"

#include <memory>

namespace qw3::detail {

class DeviceInputEmbeddingStorage final : public InputEmbeddingStorage {
public:
    DeviceInputEmbeddingStorage(std::shared_ptr<DeviceTensor> tensor,
                                uint32_t rows,
                                uint32_t dim)
        : tensor_(std::move(tensor)), rows_(rows), dim_(dim) {}

    uint32_t rows() const override { return rows_; }
    uint32_t dim() const override { return dim_; }
    const DeviceTensor &tensor() const { return *tensor_; }

private:
    std::shared_ptr<DeviceTensor> tensor_;
    uint32_t rows_ = 0;
    uint32_t dim_ = 0;
};

} // namespace qw3::detail
