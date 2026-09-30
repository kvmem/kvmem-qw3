#pragma once

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace qw3::detail {

// Retain only a suffix that could still become a stop string. A stop may span
// arbitrary token/transport boundaries; no part of a matched stop is emitted.
class CompletionStopBuffer {
public:
    explicit CompletionStopBuffer(std::vector<std::string> stops)
        : stops_(std::move(stops)) {}

    std::string push(const std::string &piece) {
        if (stopped_) return {};
        pending_ += piece;
        size_t found = std::string::npos;
        for (const auto &stop : stops_) {
            if (!stop.empty()) found = std::min(found, pending_.find(stop));
        }
        if (found != std::string::npos) {
            stopped_ = true;
            auto text = pending_.substr(0, found);
            pending_.clear();
            return text;
        }
        size_t hold = 0;
        for (const auto &stop : stops_) {
            for (size_t n = std::min(pending_.size(), stop.size()); n > hold; --n) {
                if (pending_.compare(pending_.size() - n, n, stop, 0, n) == 0) {
                    hold = n;
                    break;
                }
            }
        }
        auto text = pending_.substr(0, pending_.size() - hold);
        pending_.erase(0, pending_.size() - hold);
        return text;
    }

    std::string finish() { return std::exchange(pending_, {}); }
    bool stopped() const { return stopped_; }

private:
    std::vector<std::string> stops_;
    std::string pending_;
    bool stopped_ = false;
};

} // namespace qw3::detail
