#pragma once

#include "dflash_weights.hpp"
#include "qw3/kvmem_archive.hpp"

#include <random>
#include <vector>

namespace qw3 {

struct DFlashProposal {
    std::vector<int32_t> tokens;
    std::vector<int32_t> candidates; // [tokens.size(), top_k]
    std::vector<float> probabilities; // Conditional on the selected predecessor.
    uint32_t top_k = 0;
};

// A per-request, bounded draft cache. Only committed target features are
// appended. Speculative mask K/V are temporary and overwritten on the next
// call, so rejection cannot leave unverified target features in this cache.
class DFlashExecutor {
public:
    struct StateSnapshot {
        const DFlashWeights *weights = nullptr;
        const DeviceBackend *backend = nullptr;
        uint64_t next_position = 0;
        uint32_t rows = 0, rope_next = 0;
        std::vector<std::shared_ptr<DeviceTensor>> raw_keys, values;
    };
    DFlashExecutor(const DFlashWeights &weights, DeviceBackend &backend);
    void reset(uint64_t next_position = 0);
    StateSnapshot snapshot();
    void restore(const StateSnapshot &state);
    void capture_archive_state(KvMemArchiveState &state);
    // A legacy or different-draft archive remains usable; reset the optional
    // draft cache explicitly and report the cold start in the archive log.
    void restore_archive_state(const KvMemArchiveState &state);
    void append_context(const DeviceTensor &features, uint32_t rows, uint64_t base_position);
    const DeviceTensor &forward(const DeviceTensor &noise_embeddings, uint32_t rows,
                                uint64_t base_position);
    DFlashProposal propose(const DeviceWeight &target_head, uint32_t anchor,
                            float temperature, std::mt19937_64 &rng);
    uint32_t context_rows() const { return context_rows_; }
    uint64_t next_position() const { return next_position_; }
    uint32_t rope_next() const { return rope_next_; }
    const DFlashConfig &config() const { return weights_.config(); }

private:
    friend class DFlashBatchExecutor;
    struct Layer {
        DeviceWeight *input_norm, *ffn_norm, *q, *k, *v, *o, *q_norm, *k_norm;
        DeviceWeight *gate, *up, *down, *attn_conv, *attn_base, *mlp_conv, *mlp_base;
        std::unique_ptr<DeviceTensor> keys, values, raw_keys;
    };
    std::unique_ptr<DeviceTensor> view(DeviceTensor &buffer, uint64_t offset, uint64_t count);
    void linear(DeviceTensor &out, const DeviceWeight &weight, const DeviceTensor &input, uint32_t rows);
    void convolve(DeviceTensor &out, const DeviceTensor &input, const DeviceWeight &base,
                   uint32_t rows, uint32_t side);

    const DFlashWeights &weights_;
    DeviceBackend &backend_;
    std::vector<Layer> layers_;
    uint64_t next_position_ = 0;
    uint32_t context_rows_ = 0, rope_next_ = 0, last_rows_ = 0;
    std::unique_ptr<DeviceTensor> context_projection_, context_norm_, compact_;
    std::unique_ptr<DeviceTensor> hidden_, norm_, mixed_, delta_, q_, attention_, branch_, conv_out_;
    std::unique_ptr<DeviceTensor> ffn_gate_, ffn_up_, ffn_mid_, output_;
    std::unique_ptr<DeviceTensor> logits_, top_ids_, top_values_, top_scratch_ids_, top_scratch_values_;
    std::unique_ptr<DeviceTensor> selector_hidden_, tokens_, probabilities_, anchor_, uniforms_;
};

} // namespace qw3
