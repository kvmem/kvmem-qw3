#pragma once
#include "qw3/device_backend.hpp"
#include "qw3/qw3.hpp"
#include "dflash_executor.hpp"
#include "env_flags.hpp"
#include <cstdio>
#include <random>
#include <unordered_map>

namespace qw3::detail {
// Per-request policy/RNG bridge. Only the final 16-byte decision is copied to
// host; full logits remain available for a compatibility fallback on ties or
// numerically ambiguous cutoffs. No request RNG state is consumed on fallback.
class GpuTargetSampling {
public:
    GpuTargetSampling(DeviceBackend &backend, uint32_t vocab, const GenerationOptions &options)
        : backend_(backend), vocab_(vocab), options_(options),
          enabled_(backend.supports_target_sampling() && env_flag_enabled("QW3_GPU_TARGET_SAMPLING", true)) {}
    ~GpuTargetSampling() {
        if (calls_) std::fprintf(stderr, "[gpu-target-sampling] calls=%llu fallbacks=%llu decision_d2h_bytes=%llu\n",
            static_cast<unsigned long long>(calls_), static_cast<unsigned long long>(fallbacks_),
            static_cast<unsigned long long>(calls_ * 16));
    }
    bool enabled() const { return enabled_; }
    bool sample(const DeviceTensor *logits, const std::unordered_map<uint32_t, uint32_t> &seen,
                std::mt19937_64 &rng, DeviceSamplingResult &result,
                const std::vector<uint32_t> &drafts = {}, const DFlashProposal *proposal = nullptr,
                uint64_t row_offset = 0) {
        if (!enabled_ || !logits) return false;
        // Ragged batches store several requests in one matrix. The view owns
        // no allocation and remains alive through sample_target_rows' stream
        // synchronization; neither neighboring rows nor their RNGs are read.
        const uint64_t rows = uint64_t(drafts.size()) + 1;
        if (logits->dtype != DeviceTensorDType::F32 || logits->elem_size != sizeof(float) ||
            !vocab_ || row_offset > logits->count / vocab_ ||
            rows > logits->count / vocab_ - row_offset)
            throw std::runtime_error("GPU sampling row range mismatch");
        std::unique_ptr<DeviceTensor> view;
        if (row_offset) {
            if (!backend_.supports_tensor_views()) return false;
            view = backend_.tensor_view(const_cast<DeviceTensor &>(*logits),
                row_offset * vocab_ * sizeof(float), rows * vocab_, sizeof(float),
                DeviceTensorDType::F32, "sampling-request-rows");
            if (!view) return false;
            logits = view.get();
        }
        DeviceSamplingRequest r;
        r.vocab = vocab_; r.temperature = options_.temperature; r.top_p = options_.top_p;
        r.top_k = options_.top_k; r.min_p = options_.min_p;
        r.presence_penalty = options_.presence_penalty; r.repetition_penalty = options_.repetition_penalty;
        r.top_k_first = options_.sampling_filter_order == SamplingFilterOrder::TopKFirst;
        if (r.presence_penalty != 0 || (r.repetition_penalty > 0 && r.repetition_penalty != 1)) {
            r.seen.reserve(seen.size());
            for (const auto &entry : seen) r.seen.push_back(entry.first);
        }
        r.drafts.assign(drafts.begin(), drafts.end());
        if (proposal) {
            r.candidates_per_row = proposal->top_k;
            const size_t count = drafts.size() * proposal->top_k;
            if (!proposal->top_k || proposal->candidates.size() < count || proposal->probabilities.size() < count)
                throw std::runtime_error("GPU sampling proposal shape mismatch");
            r.candidate_ids.assign(proposal->candidates.begin(), proposal->candidates.begin() + count);
            r.proposal_probabilities.assign(proposal->probabilities.begin(), proposal->probabilities.begin() + count);
        } else if (!drafts.empty()) {
            r.residual_always_draw = false;
            r.candidates_per_row = 1; r.candidate_ids = r.drafts;
            r.proposal_probabilities.assign(drafts.size(), 1.f);
        }
        auto trial = rng;
        std::uniform_real_distribution<double> unit(0, 1);
        for (size_t i = 0; i < drafts.size() + 2; ++i) r.uniforms.push_back(unit(trial));
        const auto status = backend_.sample_target_rows(*logits, r, result);
        if (!status.ok) throw std::runtime_error(status.message);
        ++calls_;
        if (result.fallback) { ++fallbacks_; return false; }
        if (result.token < 0 || uint32_t(result.token) >= vocab_ || result.accepted < 0 ||
            size_t(result.accepted) > drafts.size() || result.rng_draws < 0 ||
            size_t(result.rng_draws) > r.uniforms.size()) throw std::runtime_error("invalid GPU sample decision");
        for (int i = 0; i < result.rng_draws; ++i) (void)unit(rng);
        return true;
    }
private:
    DeviceBackend &backend_;
    uint32_t vocab_;
    const GenerationOptions &options_;
    bool enabled_;
    uint64_t calls_ = 0, fallbacks_ = 0;
};
} // namespace qw3::detail
