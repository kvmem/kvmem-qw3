#pragma once

#include "qw3/device_backend.hpp"
#include "qw3/qw3.hpp"

#include <algorithm>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>

namespace qw3::detail {

class DeviceInputEmbeddingStorage final : public InputEmbeddingStorage {
public:
    struct Segment {
        std::shared_ptr<DeviceTensor> tensor;
        uint32_t logical_row = 0;
        uint32_t source_row = 0;
        uint32_t rows = 0;
    };

    struct ResolvedRow {
        const DeviceTensor *tensor = nullptr;
        uint32_t row = 0;
    };

    DeviceInputEmbeddingStorage(std::shared_ptr<DeviceTensor> tensor,
                                uint32_t rows,
                                uint32_t dim)
        : rows_(rows), dim_(dim) {
        segments_.push_back({std::move(tensor), 0, 0, rows});
        validate();
    }

    DeviceInputEmbeddingStorage(std::vector<Segment> segments,
                                uint32_t rows,
                                uint32_t dim)
        : segments_(std::move(segments)), rows_(rows), dim_(dim) {
        validate();
    }

    uint32_t rows() const override { return rows_; }
    uint32_t dim() const override { return dim_; }
    const DeviceTensor &tensor() const {
        if (segments_.size() != 1 || segments_[0].logical_row != 0 ||
            segments_[0].source_row != 0 || segments_[0].rows != rows_) {
            throw std::runtime_error(
                "device embedding storage is not one contiguous tensor");
        }
        return *segments_[0].tensor;
    }
    const std::vector<Segment> &segments() const { return segments_; }

    ResolvedRow resolve(uint32_t logical_row) const {
        if (logical_row >= rows_) {
            throw std::runtime_error("device embedding row is out of range");
        }
        const auto it = std::upper_bound(
            segments_.begin(), segments_.end(), logical_row,
            [](uint32_t row, const Segment &segment) {
                return row < segment.logical_row;
            });
        if (it == segments_.begin()) {
            throw std::runtime_error("device embedding row has no segment");
        }
        const Segment &segment = *std::prev(it);
        const uint32_t offset = logical_row - segment.logical_row;
        if (offset >= segment.rows) {
            throw std::runtime_error("device embedding storage has a row gap");
        }
        return {segment.tensor.get(), segment.source_row + offset};
    }

private:
    void validate() const {
        if (rows_ == 0 || dim_ == 0 || segments_.empty()) {
            throw std::runtime_error("empty device embedding storage");
        }
        uint32_t next_row = 0;
        for (const Segment &segment : segments_) {
            if (!segment.tensor || segment.rows == 0 ||
                segment.logical_row != next_row ||
                segment.source_row > segment.tensor->count / dim_ ||
                segment.rows > segment.tensor->count / dim_ - segment.source_row) {
                throw std::runtime_error("invalid device embedding segment");
            }
            next_row += segment.rows;
        }
        if (next_row != rows_) {
            throw std::runtime_error("device embedding segments do not cover rows");
        }
    }

    std::vector<Segment> segments_;
    uint32_t rows_ = 0;
    uint32_t dim_ = 0;
};

} // namespace qw3::detail
