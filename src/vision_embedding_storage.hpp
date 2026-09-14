#pragma once

#include "qw3/device_backend.hpp"
#include "qw3/qw3.hpp"

#include <algorithm>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace qw3::detail {

class DeviceInputEmbeddingStorage final : public InputEmbeddingStorage {
public:
    // Historical KV hits need only image identity and row metadata. Upload a
    // CPU-cached feature tensor only if prefill/replay actually reads its rows.
    // The owning engine keeps its backend alive throughout storage use.
    class CpuTensor {
    public:
        CpuTensor(DeviceBackend &backend,
                  std::shared_ptr<const std::vector<uint16_t>> values)
            : backend_(backend), values_(std::move(values)) {}
        uint64_t count() const { return values_ ? values_->size() : 0; }
        const DeviceTensor *get() const {
            std::call_once(once_, [&]() {
                auto tensor = backend_.tensor_bf16(count(), "vision_restored_image");
                const auto status = backend_.copy_bytes_from_host(*tensor, 0,
                    values_->data(), count() * sizeof(uint16_t));
                if (!status.ok) throw std::runtime_error(
                    std::string("restore CPU vision features: ") + status.message);
                tensor_ = std::move(tensor);
            });
            return tensor_.get();
        }
    private:
        DeviceBackend &backend_;
        std::shared_ptr<const std::vector<uint16_t>> values_;
        mutable std::once_flag once_;
        mutable std::unique_ptr<DeviceTensor> tensor_;
    };
    struct Segment {
        std::shared_ptr<DeviceTensor> tensor;
        uint32_t logical_row = 0;
        uint32_t source_row = 0;
        uint32_t rows = 0;
        std::shared_ptr<CpuTensor> cpu;
        const DeviceTensor *get() const { return tensor ? tensor.get() : cpu->get(); }
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
        return *segments_[0].get();
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
        return {segment.get(), segment.source_row + offset};
    }

private:
    void validate() const {
        if (rows_ == 0 || dim_ == 0 || segments_.empty()) {
            throw std::runtime_error("empty device embedding storage");
        }
        uint32_t next_row = 0;
        for (const Segment &segment : segments_) {
            const uint64_t count = segment.tensor ? segment.tensor->count :
                (segment.cpu ? segment.cpu->count() : 0);
            if (count == 0 || segment.rows == 0 ||
                segment.logical_row != next_row ||
                segment.source_row > count / dim_ ||
                segment.rows > count / dim_ - segment.source_row) {
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
