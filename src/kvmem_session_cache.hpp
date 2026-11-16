#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace qw3::detail {

// Optional, scheduler-owned session histories. Unlike IdleSessionCache, a
// residency limit or idle TTL only suspends execution storage; neither deletes
// history. All calls, including Active destruction, belong to one worker.
// Suspend/restore callbacks must finish outstanding transfers and provide the
// strong exception guarantee: false/throw leaves the payload in its old state.
// Callbacks must not re-enter this cache. External backend resources must outlive
// the cache and every Active, just as they must outlive the payload itself.
template<class Payload> class KvmemSessionCache {
public:
    struct Config {
        bool enabled = false;
        size_t max_histories = 0; // active + idle; exhaustion rejects a new key
        uint64_t warm_idle_ttl_ms = 0; // zero disables automatic suspension
    };
    struct Stats {
        size_t histories = 0;
        size_t active = 0;
        size_t idle_warm = 0;
        size_t idle_cold = 0;
        size_t poisoned = 0;
    };

private:
    struct Entry {
        std::string key;
        std::unique_ptr<Payload> payload;
        bool active = false;
        bool cold = false;
        bool poisoned = false;
        uint64_t idle_since_ms = 0;
    };
    struct State {
        Config config;
        // Park moves its existing node to the back without allocating. Active
        // nodes remain registered, so their key and history quota stay owned.
        std::list<Entry> entries;
        explicit State(Config value) : config(value) {}
    };
    using Iterator = typename std::list<Entry>::iterator;

public:
    class Active {
    public:
        Active() = default;
        Active(const Active &) = delete;
        Active &operator=(const Active &) = delete;
        Active(Active &&other) noexcept { move_from(other); }
        Active &operator=(Active &&other) noexcept {
            if (this != &other) {
                park(return_time_ms_);
                move_from(other);
            }
            return *this;
        }
        ~Active() { park(return_time_ms_); }

        explicit operator bool() const noexcept { return bool(state_); }
        Payload *get() const noexcept { return payload_.get(); }
        Payload *operator->() const noexcept { return get(); }
        Payload &operator*() const { return *payload_; }
        const std::string &key() const {
            if (!state_) throw std::logic_error("empty KVMem session lease");
            return entry_->key;
        }
        // A cancelled or failed controller may leave no valid continuation.
        // Keep its history and key, but require explicit erase before reuse.
        void invalidate() noexcept {
            if (state_) entry_->poisoned = true;
        }
        bool poisoned() const noexcept { return state_ && entry_->poisoned; }

        // The caller must put any borrowed executor/prefix members back into
        // the payload first. Destruction is a fallback return using the acquire
        // timestamp; normal completion supplies its current monotonic time.
        void park(uint64_t now_ms) noexcept {
            if (!state_) return;
            entry_->payload = std::move(payload_);
            entry_->active = false;
            entry_->idle_since_ms = now_ms;
            state_->entries.splice(state_->entries.end(), state_->entries, entry_);
            state_.reset();
        }

    private:
        friend class KvmemSessionCache;
        Active(std::shared_ptr<State> state, Iterator entry, uint64_t now_ms)
            : state_(std::move(state)), entry_(entry),
              payload_(std::move(entry_->payload)), return_time_ms_(now_ms) {
            entry_->active = true;
        }
        void move_from(Active &other) noexcept {
            state_ = std::move(other.state_);
            entry_ = other.entry_;
            payload_ = std::move(other.payload_);
            return_time_ms_ = other.return_time_ms_;
        }
        std::shared_ptr<State> state_;
        Iterator entry_{};
        std::unique_ptr<Payload> payload_;
        uint64_t return_time_ms_ = 0;
    };

    KvmemSessionCache() : KvmemSessionCache(Config{}) {}
    explicit KvmemSessionCache(Config config)
        : state_(std::make_shared<State>(config)) {
        if (config.enabled && !config.max_histories)
            throw std::invalid_argument("enabled KVMem session cache requires a history limit");
    }
    KvmemSessionCache(const KvmemSessionCache &) = delete;
    KvmemSessionCache &operator=(const KvmemSessionCache &) = delete;
    KvmemSessionCache(KvmemSessionCache &&) = delete;
    KvmemSessionCache &operator=(KvmemSessionCache &&) = delete;

    bool enabled() const noexcept { return state_->config.enabled; }
    size_t size() const noexcept { return state_->entries.size(); }
    bool contains(const std::string &key) const { return find(key) != state_->entries.end(); }
    const Payload *peek_idle(const std::string &key) const {
        const auto it = find(key);
        return it == state_->entries.end() || it->active ? nullptr : it->payload.get();
    }
    bool is_cold(const std::string &key) const {
        const auto it = find(key);
        return it != state_->entries.end() && it->cold;
    }
    bool is_poisoned(const std::string &key) const {
        const auto it = find(key);
        return it != state_->entries.end() && it->poisoned;
    }
    Stats stats() const {
        Stats out;
        out.histories = size();
        for (const auto &entry : state_->entries) {
            if (entry.active) ++out.active;
            else if (entry.cold) ++out.idle_cold;
            else ++out.idle_warm;
            if (entry.poisoned) ++out.poisoned;
        }
        return out;
    }

    // The factory runs only after duplicate/capacity checks. Its failure does
    // not consume history capacity or displace another history owner.
    template<class Factory>
    Active create(std::string key, Factory &&factory, uint64_t now_ms) {
        require_enabled();
        require_key(key);
        if (contains(key)) throw std::logic_error("duplicate KVMem session owner");
        if (size() >= state_->config.max_histories)
            throw std::runtime_error("KVMem session history capacity exhausted");
        std::unique_ptr<Payload> payload = factory();
        if (!payload) throw std::invalid_argument("null KVMem session payload");
        state_->entries.push_back(Entry{std::move(key), std::move(payload), false,
                                        false, false, now_ms});
        auto it = state_->entries.end();
        --it;
        return Active(state_, it, now_ms);
    }

    // A miss returns an empty lease. Busy, poisoned or failed restore is an
    // error, never a miss: callers must not rebuild over an existing lineage.
    template<class Restore>
    Active acquire(const std::string &key, Restore &&restore, uint64_t now_ms) {
        require_enabled();
        require_key(key);
        auto it = find(key);
        if (it == state_->entries.end()) return {};
        if (it->active) throw std::logic_error("KVMem session is already active");
        if (it->poisoned) throw std::logic_error("KVMem session continuation is invalid");
        if (it->cold) {
            if (!restore(*it->payload))
                throw std::runtime_error("KVMem session restore refused");
            it->cold = false;
        }
        return Active(state_, it, now_ms);
    }

    // Oldest idle owners are considered first. Refusal does not block trying
    // another owner. A thrown callback propagates; earlier successful entries
    // remain cold, while this entry remains warm. The caller must verify stats
    // against its GPU admission budget after any refused/failed suspension.
    template<class Suspend>
    size_t cool_to(size_t max_warm_idle, uint64_t now_ms, Suspend &&suspend) {
        require_enabled();
        size_t warm = stats().idle_warm;
        size_t cooled = 0;
        for (auto &entry : state_->entries) {
            if (entry.active || entry.cold) continue;
            const uint64_t ttl = state_->config.warm_idle_ttl_ms;
            const bool expired = ttl && now_ms >= entry.idle_since_ms &&
                now_ms - entry.idle_since_ms >= ttl;
            if (warm <= max_warm_idle && !expired) continue;
            if (!suspend(*entry.payload)) continue;
            entry.cold = true;
            --warm;
            ++cooled;
        }
        return cooled;
    }

    // Explicit history deletion. An active owner cannot be deleted by another
    // request; it must return to idle before administrative deletion/cleanup.
    bool erase(const std::string &key) {
        require_enabled();
        auto it = find(key);
        if (it == state_->entries.end()) return false;
        if (it->active) throw std::logic_error("cannot erase an active KVMem session");
        state_->entries.erase(it);
        return true;
    }
    void clear() {
        require_enabled();
        if (stats().active) throw std::logic_error("cannot clear active KVMem sessions");
        state_->entries.clear();
    }

private:
    Iterator find(const std::string &key) const {
        for (auto it = state_->entries.begin(); it != state_->entries.end(); ++it)
            if (it->key == key) return it;
        return state_->entries.end();
    }
    void require_enabled() const {
        if (!enabled()) throw std::logic_error("KVMem session history cache is disabled");
    }
    static void require_key(const std::string &key) {
        if (key.empty()) throw std::invalid_argument("empty KVMem session key");
    }
    std::shared_ptr<State> state_;
};

} // namespace qw3::detail
