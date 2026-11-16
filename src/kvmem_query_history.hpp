#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace qw3::detail {

// Query relevance belongs to logical input rounds, not physical KV epochs.
// The caller verifies token/media/position identity and supplies its proven
// common prefix. This class never infers identity from a block's address.
class KvMemQueryHistory {
    struct Node;

public:
    using Scores = std::unordered_map<uint64_t, double>;

    // Named session caches share immutable nodes and score maps. A snapshot
    // contains committed history only; a partially replayed input is not a
    // reusable observation. Copying/restoring one does not copy score vectors.
    class Snapshot {
    public:
        bool empty() const noexcept { return !tip_; }

    private:
        friend class KvMemQueryHistory;
        std::shared_ptr<const Node> tip_;
    };

    // input_end is the accepted input identity boundary, excluding a temporary
    // assistant generation header when the caller can prove that boundary.
    // Strictly excluding input_end from parent selection makes regeneration
    // recompute this input from its parent instead of rewarding it twice.
    void begin(uint64_t input_end, uint64_t verified_prefix, double decay) {
        if (!std::isfinite(decay) || decay < 0 || decay > 1)
            throw std::invalid_argument("KVMem query history decay must be in [0, 1]");

        auto parent = tip_;
        while (parent && (parent->input_end > verified_prefix ||
                          parent->input_end >= input_end))
            parent = parent->parent;

        parent_ = std::move(parent);
        candidate_ = parent_ ? parent_->scores : nullptr;
        input_end_ = input_end;
        decay_ = decay;
        observed_ = false;
        open_ = true;
    }

    // raw contains nonnegative query relevance keyed by logical block start.
    // The caller defines the eligible scoring domain by the keys it supplies;
    // starts at/beyond the accepted input boundary never enter the normalizer or
    // committed history. An old block absent from raw retains decayed history.
    // Every preview replaces its predecessor from the SAME immutable parent.
    const Scores &preview(const Scores &raw) {
        if (!open_)
            throw std::logic_error("KVMem query history preview requires an open round");

        double maximum = 0;
        for (const auto &[start, value] : raw) {
            if (!std::isfinite(value) || value < 0)
                throw std::invalid_argument("KVMem query history scores must be finite and nonnegative");
            if (start < input_end_) maximum = std::max(maximum, value);
        }
        if (maximum == 0) {
            candidate_ = parent_ ? parent_->scores : nullptr;
            observed_ = false;
            return scores();
        }

        // Scaling before summation handles finite values whose direct sum
        // would overflow, and preserves a valid very small nonzero signal.
        long double total = 0;
        for (const auto &[start, value] : raw)
            if (start < input_end_)
                total += static_cast<long double>(value) / maximum;

        auto next = std::make_shared<Scores>();
        if (parent_ && parent_->scores && decay_ > 0) {
            next->reserve(parent_->scores->size());
            for (const auto &[start, value] : *parent_->scores) {
                const double retained = decay_ * value;
                if (start < input_end_ && retained > 0)
                    next->emplace(start, retained);
            }
        }
        for (const auto &[start, value] : raw) {
            if (start >= input_end_ || value == 0) continue;
            const double normalized = static_cast<double>(
                (static_cast<long double>(value) / maximum) / total);
            if (normalized > 0) (*next)[start] += normalized;
        }
        candidate_ = std::move(next);
        observed_ = true;
        return scores();
    }

    // A no-observation input records a boundary while sharing the unchanged
    // parent map: a cold/zero query must not silently age the old evidence.
    // Replacing the tip drops this object's abandoned future branch; snapshots
    // still retain their own branch. Repeated commits do nothing.
    bool commit() {
        if (!open_) return false;
        auto next = std::make_shared<Node>();
        next->parent = parent_;
        next->scores = candidate_;
        next->input_end = input_end_;
        next->round = parent_ ? parent_->round + 1 : 1;
        tip_ = std::move(next);
        parent_.reset();
        candidate_.reset();
        open_ = false;
        return true;
    }

    // No committed state changes until commit, including when beginning a
    // divergent input. Aborting therefore restores the complete prior branch.
    void abort() noexcept {
        parent_.reset();
        candidate_.reset();
        observed_ = false;
        open_ = false;
    }

    const Scores &scores() const noexcept {
        return values(open_ ? candidate_ : (tip_ ? tip_->scores : nullptr));
    }

    const Scores &committed_scores() const noexcept {
        return values(tip_ ? tip_->scores : nullptr);
    }

    bool open() const noexcept { return open_; }
    bool has_observation() const noexcept { return observed_; }
    uint64_t input_end() const noexcept {
        return open_ ? input_end_ : (tip_ ? tip_->input_end : 0);
    }
    uint64_t round_count() const noexcept { return tip_ ? tip_->round : 0; }

    Snapshot snapshot() const noexcept {
        Snapshot saved;
        saved.tip_ = tip_;
        return saved;
    }

    void restore(const Snapshot &saved) noexcept {
        abort();
        tip_ = saved.tip_;
    }

    void clear() noexcept {
        abort();
        tip_.reset();
    }

private:
    struct Node {
        std::shared_ptr<const Node> parent;
        std::shared_ptr<const Scores> scores;
        uint64_t input_end = 0;
        uint64_t round = 0;
    };

    static const Scores &values(const std::shared_ptr<const Scores> &scores) noexcept {
        static const Scores empty;
        return scores ? *scores : empty;
    }

    std::shared_ptr<const Node> tip_;
    std::shared_ptr<const Node> parent_;
    std::shared_ptr<const Scores> candidate_;
    uint64_t input_end_ = 0;
    double decay_ = .5;
    bool open_ = false;
    bool observed_ = false;
};

} // namespace qw3::detail
