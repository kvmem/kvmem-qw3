#pragma once

#include "dflash_executor.hpp"

namespace qw3 {

// Joint draft execution for requests with the same proposal width and
// temperature. Committed context and logical/RoPE state remain per-request.
class DFlashBatchExecutor {
public:
    struct Request {
        DFlashExecutor *executor = nullptr;
        const DeviceTensor *noise = nullptr;
        uint32_t rows = 0;
        uint64_t base_position = 0;
        uint32_t anchor = 0;
        float temperature = 0;
        std::mt19937_64 *rng = nullptr;
    };
    DFlashBatchExecutor(const DFlashWeights &weights, DeviceBackend &backend)
        : weights_(weights), backend_(backend) {}
    std::vector<DFlashProposal> propose(const std::vector<Request> &requests,
                                         const DeviceWeight &target_head);
    // Valid rows are the concatenated blocks from the last completed call.
    const DeviceTensor *output() const { return output_.get(); }
private:
    void reserve(uint32_t rows, uint32_t blocks);
    std::unique_ptr<DeviceTensor> view(DeviceTensor &, uint64_t offset, uint64_t count);
    void linear(DeviceTensor &, const DeviceWeight &, const DeviceTensor &, uint32_t rows);
    const DFlashWeights &weights_;
    DeviceBackend &backend_;
    uint32_t capacity_ = 0, blocks_capacity_ = 0, active_blocks_ = 0;
    std::unique_ptr<DeviceTensor> hidden_, norm_, mixed_, delta_, q_, k_, v_, attention_, branch_, conv_out_;
    std::unique_ptr<DeviceTensor> gate_, up_, mid_, output_, proposal_hidden_, metadata_;
    std::unique_ptr<DeviceTensor> logits_, top_ids_, top_values_, scratch_ids_, scratch_values_;
    std::unique_ptr<DeviceTensor> selector_, tokens_, probabilities_, anchors_, uniforms_;
};
} // namespace qw3
