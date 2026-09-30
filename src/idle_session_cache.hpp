#pragma once

#include <cstddef>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace qw3::detail {

// Worker-owned idle residents only. Active requests hold their payload outside
// this cache, so eviction cannot destroy an executor used by a controller.
// The scheduler supplies its free-slot budget before creating another resident.
template<class Payload> class IdleSessionCache {
public:
    std::unique_ptr<Payload> take(const std::string &key) {
        for (auto it = idle_.begin(); it != idle_.end(); ++it) {
            if (it->first != key) continue;
            auto value = std::move(it->second);
            idle_.erase(it);
            return value;
        }
        return {};
    }

    void park(std::string key, std::unique_ptr<Payload> value) {
        if (key.empty() || !value) throw std::invalid_argument("invalid idle session");
        for (const auto &entry : idle_)
            if (entry.first == key) throw std::logic_error("duplicate idle session owner");
        idle_.emplace_back(std::move(key), std::move(value));
    }

    size_t trim(size_t max_idle) {
        size_t evicted = 0;
        while (idle_.size() > max_idle) { idle_.pop_front(); ++evicted; }
        return evicted;
    }
    size_t size() const { return idle_.size(); }
    void clear() { idle_.clear(); }

private:
    std::list<std::pair<std::string, std::unique_ptr<Payload>>> idle_;
};

} // namespace qw3::detail
