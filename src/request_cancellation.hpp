#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace qw3::detail {

// The lease spans preprocessing, queued work, GPU teardown and streaming.
// A cancellation acknowledgement is complete only after the last lease dies.
class RequestCancellationRegistry {
public:
    struct State {
        std::atomic<bool> cancelled{false};
        std::atomic<bool> completed{false};
    };
    struct Lease {
        std::shared_ptr<State> state;
        ~Lease() { state->completed.store(true); }
    };
    std::shared_ptr<Lease> begin(const std::string &id) {
        auto state = std::make_shared<State>();
        if (!id.empty()) {
            if (id.size() > 128 || id.find_first_not_of(
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.") != std::string::npos) {
                throw std::runtime_error("invalid x-qw3-request-id");
            }
            std::lock_guard<std::mutex> lock(mu_);
            // Keep a bounded history of acknowledgements. Never discard live work.
            if (states_.size() >= 4096) {
                for (auto it = states_.begin(); it != states_.end();) {
                    if (it->second->completed.load()) it = states_.erase(it);
                    else ++it;
                }
            }
            if (states_.count(id)) throw std::runtime_error("duplicate x-qw3-request-id");
            states_[id] = state;
        }
        auto lease = std::make_shared<Lease>();
        lease->state = std::move(state);
        return lease;
    }
    std::shared_ptr<State> cancel(const std::string &id) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = states_.find(id);
        if (it == states_.end()) return {};
        it->second->cancelled.store(true);
        return it->second;
    }
private:
    std::mutex mu_;
    std::unordered_map<std::string, std::shared_ptr<State>> states_;
};
} // namespace qw3::detail
